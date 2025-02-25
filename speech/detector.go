package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"io"
	"log"
	"log/slog"
	"sync"
	"unsafe"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

const (
	stateLen   = 2 * 1 * 128
	contextLen = 64
)

type LogLevel int

func (l LogLevel) OrtLoggingLevel() C.OrtLoggingLevel {
	switch l {
	case LevelVerbose:
		return C.ORT_LOGGING_LEVEL_VERBOSE
	case LogLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO
	case LogLevelWarn:
		return C.ORT_LOGGING_LEVEL_WARNING
	case LogLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR
	case LogLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL
	default:
		return C.ORT_LOGGING_LEVEL_WARNING
	}
}

const (
	LevelVerbose LogLevel = iota + 1
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

type DetectorConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence to wait for each speech segment before separating it.
	MinSilenceDurationMs int
	// The maximum duration of speech chunks in seconds
	MaxSpeechDurationS int
	// The padding to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
	// The loglevel for the onnx environment, by default it is set to LogLevelWarn.
	LogLevel LogLevel
}

func (c DetectorConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if c.SampleRate != 8000 && c.SampleRate != 16000 {
		return fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if c.Threshold <= 0 || c.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}

	if c.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be a positive number")
	}

	if c.MaxSpeechDurationS < 0 {
		return fmt.Errorf("invalid MaxSpeechDurationS: should be a positive number")
	}

	if c.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be a positive number")
	}

	return nil
}

type Detector struct {
	api         *C.OrtApi
	env         *C.OrtEnv
	sessionOpts *C.OrtSessionOptions
	session     *C.OrtSession
	memoryInfo  *C.OrtMemoryInfo
	cStrings    map[string]*C.char

	cfg DetectorConfig

	state [stateLen]float32
	ctx   [contextLen]float32

	currSample int
	triggered  bool
	tempEnd    int
}

func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	sd := Detector{
		cfg:      cfg,
		cStrings: map[string]*C.char{},
	}

	sd.api = C.OrtGetApi()
	if sd.api == nil {
		return nil, fmt.Errorf("failed to get API")
	}

	sd.cStrings["loggerName"] = C.CString("vad")
	status := C.OrtApiCreateEnv(sd.api, cfg.LogLevel.OrtLoggingLevel(), sd.cStrings["loggerName"], &sd.env)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create env: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateSessionOptions(sd.api, &sd.sessionOpts)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session options: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetIntraOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set intra threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetInterOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set inter threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetSessionGraphOptimizationLevel(sd.api, sd.sessionOpts, C.ORT_ENABLE_ALL)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set session graph optimization level: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	sd.cStrings["modelPath"] = C.CString(sd.cfg.ModelPath)
	status = C.OrtApiCreateSession(sd.api, sd.env, sd.cStrings["modelPath"], sd.sessionOpts, &sd.session)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateCpuMemoryInfo(sd.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &sd.memoryInfo)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	sd.cStrings["input"] = C.CString("input")
	sd.cStrings["sr"] = C.CString("sr")
	sd.cStrings["state"] = C.CString("state")
	sd.cStrings["stateN"] = C.CString("stateN")
	sd.cStrings["output"] = C.CString("output")

	return &sd, nil
}

type SafeWavDecoder struct {
	*wav.Decoder
	mu sync.Mutex
}

func (dec *SafeWavDecoder) GetSamples(idxStart int, idxEnd int) []float32 {
	dec.mu.Lock()
	defer dec.mu.Unlock()

	// Save the current position of the cursor
	// so that we can go back to it after getting the samples
	prevOffset, err := dec.Seek(0, io.SeekCurrent)
	if err != nil {
		log.Printf("error reading previous seek position: %w", err)
	}

	dec.Rewind()
	dec.Seek(int64(idxStart)*2, io.SeekCurrent) // *2 because of the 16-bit pcm encoding (2 bytes per value)

	intBuf := &audio.IntBuffer{Data: make([]int, idxEnd-idxStart)}
	n, err := dec.PCMBuffer(intBuf)
	if err != nil {
		log.Printf("Length of PCM: %d\n", dec.PCMSize)
		log.Printf("Requested - start: %d, end: %d\n", idxStart, idxEnd)
		log.Printf("error reading PCM buffer: %w", err)
	}
	data := intBuf.AsFloat32Buffer().Data[:n]

	// Return the cursor to the previous position
	dec.Seek(prevOffset, io.SeekStart)

	return data
}

func (dec *SafeWavDecoder) ReadPCMBuffer(buf *audio.IntBuffer) (int, error) {
	dec.mu.Lock()
	defer dec.mu.Unlock()

	n, err := dec.PCMBuffer(buf)

	return n, err
}

// Segment contains timing information of a speech segment.
type Segment struct {
	// The relative timestamp in samples of when a speech segment begins.
	SpeechStartAt int
	// The relative timestamp in samples of when a speech segment ends.
	SpeechEndAt int
}

func (sd *Detector) Detect(dec *SafeWavDecoder) (<-chan Segment, <-chan error, <-chan bool) {
	segmentCh := make(chan Segment, 1)
	errorCh := make(chan error, 1)
	doneCh := make(chan bool, 1)

	if sd == nil {
		errorCh <- fmt.Errorf("invalid nil detector")
	}

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	slog.Debug("starting speech detection")

	minSilenceSamples := sd.cfg.MinSilenceDurationMs * sd.cfg.SampleRate / 1000
	speechPadSamples := sd.cfg.SpeechPadMs * sd.cfg.SampleRate / 1000
	maxSpeechSamples := sd.cfg.MaxSpeechDurationS*sd.cfg.SampleRate - windowSize - (2 * speechPadSamples)
	minSpeechSamples := 250 * sd.cfg.SampleRate / 1000

	var segment Segment
	var speechStartAt, speechEndAt int
	var prevEnd, nextStart int

	go func() {
		if ok := dec.IsValidFile(); !ok {
			errorCh <- fmt.Errorf("invalid WAV file")
		}
		i := 0

	InferenceLoop:
		for {
			buffer := &audio.IntBuffer{Data: make([]int, windowSize)}
			n, err := dec.ReadPCMBuffer(buffer)
			if err != nil {
				errorCh <- fmt.Errorf("error reading PCM buffer: %w", err)
			}
			if n < windowSize {
				break InferenceLoop
			}
			pcmData := buffer.AsFloat32Buffer().Data

			speechProb, err := sd.infer(pcmData)
			if err != nil {
				errorCh <- fmt.Errorf("infer failed: %w", err)
			}

			sd.currSample += windowSize

			if speechProb >= sd.cfg.Threshold && sd.tempEnd != 0 {
				sd.tempEnd = 0
				if nextStart < prevEnd {
					nextStart = i
				}
			}

			if speechProb >= sd.cfg.Threshold && !sd.triggered {
				sd.triggered = true
				speechStartAt = i - windowSize - speechPadSamples

				// We clamp at zero since due to padding the starting position could be negative.
				if speechStartAt < 0 {
					speechStartAt = 0
				}

				slog.Debug("speech start", slog.Int("startAt", speechStartAt))
				segment.SpeechStartAt = speechStartAt
				i += windowSize
				continue
			}

			if sd.triggered && (i-speechStartAt > maxSpeechSamples) {
				if prevEnd != 0 {
					segment.SpeechEndAt = prevEnd
					segmentCh <- segment
					if nextStart < prevEnd {
						sd.triggered = false
					} else {
						segment.SpeechStartAt = nextStart
					}
					prevEnd, nextStart, sd.tempEnd = 0, 0, 0
				} else {
					segment.SpeechEndAt = i
					segmentCh <- segment
					prevEnd, nextStart, sd.tempEnd = 0, 0, 0
					sd.triggered = false
					i += windowSize
					continue
				}
			}

			if speechProb < (sd.cfg.Threshold-0.15) && sd.triggered {
				if sd.tempEnd == 0 {
					sd.tempEnd = i
				}

				// Not enough silence yet to split, we continue.
				if i-sd.tempEnd < minSilenceSamples {
					i += windowSize
					continue
				}

				segment.SpeechEndAt = sd.tempEnd + speechPadSamples
				prevEnd, nextStart, sd.tempEnd = 0, 0, 0
				sd.triggered = false
				slog.Debug("speech end", slog.Int("endAt", speechEndAt))

				if segment.SpeechEndAt-segment.SpeechStartAt > minSpeechSamples {
					segmentCh <- segment
				}
			}

			i += windowSize
		}
		slog.Debug("speech detection done")
		close(doneCh)
	}()

	return segmentCh, errorCh, doneCh
}

func (sd *Detector) Reset() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	sd.currSample = 0
	sd.triggered = false
	sd.tempEnd = 0
	for i := 0; i < stateLen; i++ {
		sd.state[i] = 0
	}
	for i := 0; i < contextLen; i++ {
		sd.ctx[i] = 0
	}

	return nil
}

func (sd *Detector) SetThreshold(value float32) {
	sd.cfg.Threshold = value
}

func (sd *Detector) Destroy() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	C.OrtApiReleaseMemoryInfo(sd.api, sd.memoryInfo)
	C.OrtApiReleaseSession(sd.api, sd.session)
	C.OrtApiReleaseSessionOptions(sd.api, sd.sessionOpts)
	C.OrtApiReleaseEnv(sd.api, sd.env)
	for _, ptr := range sd.cStrings {
		C.free(unsafe.Pointer(ptr))
	}

	return nil
}
