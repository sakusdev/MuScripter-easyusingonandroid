package dev.sakus.muscriptoreasy.inference

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Kotlin port of MuScriptor's `utils.resample.ResampleFrac` (Julius algorithm).
 *
 * Defaults intentionally match upstream: zeros=24, rolloff=0.945. The source
 * implementation performs a bank of FIR convolutions with stride old_sr after
 * reducing both sample rates by their GCD; this class computes the same
 * polyphase outputs directly without materializing a padded signal tensor.
 */
class SincResampler(
    oldSampleRate: Int,
    newSampleRate: Int,
    private val zeros: Int = 24,
    private val rolloff: Float = 0.945f,
) {
    private val oldRate: Int
    private val newRate: Int
    private val width: Int
    private val kernels: Array<FloatArray>

    init {
        require(oldSampleRate > 0 && newSampleRate > 0) { "Sample rates must be positive" }
        val gcd = gcd(oldSampleRate, newSampleRate)
        oldRate = oldSampleRate / gcd
        newRate = newSampleRate / gcd

        if (oldRate == newRate) {
            width = 0
            kernels = emptyArray()
        } else {
            val effectiveRate = min(oldRate, newRate) * rolloff
            width = ceil(zeros * oldRate / effectiveRate).toInt()
            val kernelLength = 2 * width + oldRate
            kernels = Array(newRate) { phase ->
                val kernel = FloatArray(kernelLength)
                var sum = 0f
                for (k in 0 until kernelLength) {
                    val idx = -width + k
                    var t = (
                        -phase.toFloat() / newRate + idx.toFloat() / oldRate
                    ) * effectiveRate
                    t = max(-zeros.toFloat(), min(zeros.toFloat(), t))
                    t *= PI.toFloat()

                    val windowBase = cos((t / zeros / 2f).toDouble()).toFloat()
                    val window = windowBase * windowBase
                    val sinc = if (t == 0f) 1f else sin(t.toDouble()).toFloat() / t
                    val value = sinc * window
                    kernel[k] = value
                    sum += value
                }
                require(sum != 0f) { "Invalid zero-sum resampling kernel" }
                for (k in kernel.indices) kernel[k] /= sum
                kernel
            }
        }
    }

    fun process(input: FloatArray): FloatArray {
        if (oldRate == newRate) return input.copyOf()
        if (input.isEmpty()) return FloatArray(0)

        val outputLength = floor(newRate.toDouble() * input.size / oldRate).toInt()
        val output = FloatArray(outputLength)

        for (outIndex in 0 until outputLength) {
            val frame = outIndex / newRate
            val phase = outIndex - frame * newRate
            val kernel = kernels[phase]
            val paddedStart = frame * oldRate
            var sum = 0f
            for (k in kernel.indices) {
                // F.pad(..., mode="replicate") from upstream.
                val sourceIndex = (paddedStart + k - width).coerceIn(0, input.lastIndex)
                sum += input[sourceIndex] * kernel[k]
            }
            output[outIndex] = sum
        }
        return output
    }

    companion object {
        private fun gcd(a0: Int, b0: Int): Int {
            var a = a0
            var b = b0
            while (b != 0) {
                val t = a % b
                a = b
                b = t
            }
            return a
        }
    }
}
