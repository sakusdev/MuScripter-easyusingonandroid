package dev.sakus.muscriptoreasy.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.ln
import kotlin.math.sin

class MuScriptorLogMelTest {
    private val frontend = MuScriptorLogMel()

    @Test
    fun zeroAudioMatchesLogEpsilon() {
        val out = frontend.compute(FloatArray(MuScriptorLogMel.SEGMENT_SAMPLES))
        assertEquals(MuScriptorLogMel.FRAMES * MuScriptorLogMel.MEL_BINS, out.size)
        val expected = ln(1e-6).toFloat()
        for (index in intArrayOf(0, 1, 511, 512, out.lastIndex)) {
            assertEquals(expected, out[index], 1e-5f)
        }
    }

    @Test
    fun sine440HzMatchesUpstreamReferenceNearMiddleOfChunk() {
        val samples = FloatArray(MuScriptorLogMel.SEGMENT_SAMPLES) { i ->
            sin(2.0 * PI * 440.0 * i / MuScriptorLogMel.SAMPLE_RATE).toFloat()
        }
        val out = frontend.compute(samples)
        val frame = 250
        val base = frame * MuScriptorLogMel.MEL_BINS

        var peak = 0
        var peakValue = Float.NEGATIVE_INFINITY
        for (mel in 0 until MuScriptorLogMel.MEL_BINS) {
            val value = out[base + mel]
            if (value > peakValue) {
                peakValue = value
                peak = mel
            }
        }

        // Generated with upstream pure-torch _MelSpectrogram on float32 audio:
        // frame 250 -> peak mel 98, log value 5.991836.
        assertEquals(98, peak)
        assertEquals(5.991836f, peakValue, 0.03f)
        assertEquals(4.938666f, out[base + 97], 0.04f)
        assertEquals(5.671216f, out[base + 99], 0.04f)
        assertTrue(out.all { it.isFinite() })
    }
}
