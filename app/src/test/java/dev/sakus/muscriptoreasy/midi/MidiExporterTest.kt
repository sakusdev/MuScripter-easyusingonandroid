package dev.sakus.muscriptoreasy.midi

import dev.sakus.muscriptoreasy.inference.DecodedNoteEvent
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MidiExporterTest {
    @Test
    fun writesType1HeaderTempoAndPianoTrack() {
        val midi = MidiExporter.encode(
            listOf(
                DecodedNoteEvent.Start(program = 0, pitch = 60, timeSeconds = 0.0),
                DecodedNoteEvent.End(program = 0, pitch = 60, timeSeconds = 0.5),
            ),
        )

        assertArrayEquals("MThd".toByteArray(), midi.copyOfRange(0, 4))
        assertEquals(6, int32(midi, 4))
        assertEquals(1, int16(midi, 8))
        assertEquals(2, int16(midi, 10)) // conductor + piano
        assertEquals(480, int16(midi, 12))

        assertTrue(contains(midi, byteArrayOf(0xFF.toByte(), 0x51, 0x03, 0x07, 0xA1.toByte(), 0x20)))
        assertTrue(contains(midi, byteArrayOf(0xC0.toByte(), 0x00)))
        assertTrue(contains(midi, byteArrayOf(0x90.toByte(), 60, 100)))
        // 0.5 sec at 120 BPM / PPQ 480 = 480 ticks = VLQ 83 60.
        assertTrue(contains(midi, byteArrayOf(0x83.toByte(), 0x60, 0x80.toByte(), 60, 0)))
    }

    @Test
    fun drumsUseChannelNineAndOwnTrack() {
        val midi = MidiExporter.encode(
            listOf(
                DecodedNoteEvent.Start(128, 36, 0.25, isDrum = true),
                DecodedNoteEvent.End(128, 36, 0.26, isDrum = true),
            ),
        )

        assertEquals(2, int16(midi, 10))
        assertTrue(contains(midi, "drums".toByteArray()))
        assertTrue(contains(midi, byteArrayOf(0xC9.toByte(), 0x00)))
        assertTrue(contains(midi, byteArrayOf(0x99.toByte(), 36, 100)))
        assertTrue(contains(midi, byteArrayOf(0x89.toByte(), 36, 0)))
    }

    @Test
    fun differentProgramsReceiveDifferentMelodicChannels() {
        val midi = MidiExporter.encode(
            listOf(
                DecodedNoteEvent.Start(0, 60, 0.0),
                DecodedNoteEvent.End(0, 60, 0.1),
                DecodedNoteEvent.Start(24, 64, 0.2),
                DecodedNoteEvent.End(24, 64, 0.3),
            ),
        )

        assertEquals(3, int16(midi, 10))
        assertTrue(contains(midi, byteArrayOf(0xC0.toByte(), 0x00)))
        assertTrue(contains(midi, byteArrayOf(0xC1.toByte(), 24)))
        assertTrue(contains(midi, byteArrayOf(0x91.toByte(), 64, 100)))
    }

    private fun int16(data: ByteArray, offset: Int): Int =
        ((data[offset].toInt() and 0xff) shl 8) or (data[offset + 1].toInt() and 0xff)

    private fun int32(data: ByteArray, offset: Int): Int =
        ((data[offset].toInt() and 0xff) shl 24) or
            ((data[offset + 1].toInt() and 0xff) shl 16) or
            ((data[offset + 2].toInt() and 0xff) shl 8) or
            (data[offset + 3].toInt() and 0xff)

    private fun contains(haystack: ByteArray, needle: ByteArray): Boolean {
        if (needle.isEmpty()) return true
        if (needle.size > haystack.size) return false
        for (i in 0..haystack.size - needle.size) {
            var match = true
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) {
                    match = false
                    break
                }
            }
            if (match) return true
        }
        return false
    }
}
