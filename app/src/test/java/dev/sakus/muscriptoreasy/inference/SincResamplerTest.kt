package dev.sakus.muscriptoreasy.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SincResamplerTest {
    @Test
    fun identityReturnsCopy() {
        val input = floatArrayOf(-1f, 0f, 0.25f, 1f)
        val output = SincResampler(16_000, 16_000).process(input)
        assertArrayEquals(input, output, 0f)
    }

    @Test
    fun fourToFiveMatchesUpstreamJuliusReference() {
        val input = FloatArray(10) { it.toFloat() }
        val output = SincResampler(4, 5).process(input)
        val expected = floatArrayOf(
            0.020223964f,
            0.742638946f,
            1.643618345f,
            2.384717226f,
            3.189271927f,
            4.022982597f,
            4.783051490f,
            5.596724510f,
            6.426414490f,
            7.162165165f,
            8.021217346f,
            8.855781555f,
        )
        assertEquals(expected.size, output.size)
        assertArrayEquals(expected, output, 2e-5f)
    }
}
