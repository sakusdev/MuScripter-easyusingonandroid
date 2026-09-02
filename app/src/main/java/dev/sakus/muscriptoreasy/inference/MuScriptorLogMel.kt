package dev.sakus.muscriptoreasy.inference

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin reproduction of MuScriptor's audio frontend.
 *
 * Upstream parameters:
 * - 16 kHz mono, 5 s / 80,000 samples
 * - n_fft = win_length = 2048
 * - hop = 160 (100 Hz)
 * - periodic Hann window
 * - center=true, reflect padding
 * - onesided, unnormalized magnitude STFT (power=1)
 * - 512-bin HTK mel filterbank, norm=null, 0..8 kHz
 * - log(mel + 1e-6)
 *
 * The returned FloatArray is row-major [501, 512], ready to wrap as an
 * ExecuTorch tensor with shape [1, 501, 512].
 */
class MuScriptorLogMel {
    private val hann = FloatArray(N_FFT) { n ->
        (0.5 - 0.5 * cos(2.0 * PI * n / N_FFT)).toFloat()
    }

    private val twiddleReal = FloatArray(N_FFT / 2) { k ->
        cos(-2.0 * PI * k / N_FFT).toFloat()
    }
    private val twiddleImag = FloatArray(N_FFT / 2) { k ->
        sin(-2.0 * PI * k / N_FFT).toFloat()
    }

    private data class SparseMelBand(
        val firstBin: Int,
        val weights: FloatArray,
    )

    private val melBands: Array<SparseMelBand> = buildMelBands()

    fun compute(input: FloatArray): FloatArray {
        require(input.size <= SEGMENT_SAMPLES) {
            "Expected at most $SEGMENT_SAMPLES samples, got ${input.size}"
        }

        // Upstream zero-pads the final short 5-second chunk before running STFT.
        val segment = FloatArray(SEGMENT_SAMPLES)
        input.copyInto(segment)

        // torch.stft(center=true, pad_mode="reflect") pads n_fft/2 samples.
        val padded = FloatArray(SEGMENT_SAMPLES + 2 * CENTER_PAD)
        segment.copyInto(padded, destinationOffset = CENTER_PAD)
        for (i in 0 until CENTER_PAD) {
            padded[CENTER_PAD - 1 - i] = segment[i + 1]
            padded[CENTER_PAD + SEGMENT_SAMPLES + i] = segment[SEGMENT_SAMPLES - 2 - i]
        }

        val output = FloatArray(FRAMES * MEL_BINS)
        val real = FloatArray(N_FFT)
        val imag = FloatArray(N_FFT)
        val magnitude = FloatArray(FREQ_BINS)

        for (frame in 0 until FRAMES) {
            val start = frame * HOP_LENGTH
            for (n in 0 until N_FFT) {
                real[n] = padded[start + n] * hann[n]
                imag[n] = 0f
            }

            fftInPlace(real, imag)

            for (bin in 0 until FREQ_BINS) {
                val re = real[bin]
                val im = imag[bin]
                magnitude[bin] = sqrt((re * re + im * im).toDouble()).toFloat()
            }

            val outBase = frame * MEL_BINS
            for (mel in 0 until MEL_BINS) {
                val band = melBands[mel]
                var sum = 0f
                var bin = band.firstBin
                for (weight in band.weights) {
                    sum += magnitude[bin] * weight
                    bin++
                }
                output[outBase + mel] = ln((sum + LOG_EPS).toDouble()).toFloat()
            }
        }
        return output
    }

    /** Iterative radix-2 Cooley-Tukey FFT, matching torch.stft's no-normalization convention. */
    private fun fftInPlace(real: FloatArray, imag: FloatArray) {
        var j = 0
        for (i in 1 until N_FFT) {
            var bit = N_FFT shr 1
            while ((j and bit) != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
        }

        var length = 2
        while (length <= N_FFT) {
            val half = length shr 1
            val twiddleStep = N_FFT / length
            var base = 0
            while (base < N_FFT) {
                for (offset in 0 until half) {
                    val twiddle = offset * twiddleStep
                    val wr = twiddleReal[twiddle]
                    val wi = twiddleImag[twiddle]
                    val even = base + offset
                    val odd = even + half

                    val or = real[odd]
                    val oi = imag[odd]
                    val vr = or * wr - oi * wi
                    val vi = or * wi + oi * wr
                    val ur = real[even]
                    val ui = imag[even]

                    real[even] = ur + vr
                    imag[even] = ui + vi
                    real[odd] = ur - vr
                    imag[odd] = ui - vi
                }
                base += length
            }
            length = length shl 1
        }
    }

    private fun buildMelBands(): Array<SparseMelBand> {
        val melMax = hzToMel(SAMPLE_RATE / 2f)
        val melPoints = FloatArray(MEL_BINS + 2) { i ->
            (melMax * i / (MEL_BINS + 1)).toFloat()
        }
        val hzPoints = FloatArray(MEL_BINS + 2) { i -> melToHz(melPoints[i]) }
        val binHz = (SAMPLE_RATE / 2f) / (FREQ_BINS - 1)

        return Array(MEL_BINS) { mel ->
            val lower = hzPoints[mel]
            val center = hzPoints[mel + 1]
            val upper = hzPoints[mel + 2]
            var first = -1
            var last = -1
            val dense = FloatArray(FREQ_BINS)

            for (bin in 0 until FREQ_BINS) {
                val freq = bin * binHz
                val down = (freq - lower) / (center - lower)
                val up = (upper - freq) / (upper - center)
                val weight = max(0f, min(down, up))
                dense[bin] = weight
                if (weight > 0f) {
                    if (first < 0) first = bin
                    last = bin
                }
            }

            if (first < 0) {
                SparseMelBand(0, FloatArray(0))
            } else {
                SparseMelBand(first, dense.copyOfRange(first, last + 1))
            }
        }
    }

    private fun hzToMel(freq: Float): Float =
        (2595.0 * log10(1.0 + freq / 700.0)).toFloat()

    private fun melToHz(mel: Float): Float =
        (700.0 * (10.0.pow(mel / 2595.0) - 1.0)).toFloat()

    companion object {
        const val SAMPLE_RATE = 16_000
        const val SEGMENT_SAMPLES = 80_000
        const val N_FFT = 2_048
        const val CENTER_PAD = N_FFT / 2
        const val HOP_LENGTH = 160
        const val FREQ_BINS = N_FFT / 2 + 1
        const val MEL_BINS = 512
        const val FRAMES = SEGMENT_SAMPLES / HOP_LENGTH + 1
        const val LOG_EPS = 1e-6f
    }
}
