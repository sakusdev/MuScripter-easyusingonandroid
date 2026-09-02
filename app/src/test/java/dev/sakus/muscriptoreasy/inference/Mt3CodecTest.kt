package dev.sakus.muscriptoreasy.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class Mt3CodecTest {
    @Test
    fun vocabularyLayoutMatchesMuScriptor() {
        assertEquals(1004, Mt3Vocab.PITCH_BASE)
        assertEquals(1132, Mt3Vocab.VELOCITY_BASE)
        assertEquals(1134, Mt3Vocab.TIE)
        assertEquals(1135, Mt3Vocab.PROGRAM_BASE)
        assertEquals(1265, Mt3Vocab.DRUM_BASE)
        assertEquals(1393, Mt3Vocab.GENERATED_VOCAB_SIZE)
    }

    @Test
    fun decodesSimpleNoteAtCentisecondTiming() {
        val decoder = Mt3StreamDecoder()
        decoder.beginChunk(0.0, 5.0)
        decoder.feed(Mt3Vocab.TIE)
        decoder.feed(Mt3Vocab.programToken(0))
        decoder.feed(Mt3Vocab.VELOCITY_BASE + 1)
        decoder.feed(Mt3Vocab.SHIFT_BASE + 16)

        val start = decoder.feed(Mt3Vocab.pitchToken(60)).single() as DecodedNoteEvent.Start
        assertEquals(60, start.pitch)
        assertEquals(0.16, start.timeSeconds, 1e-9)

        decoder.feed(Mt3Vocab.VELOCITY_BASE)
        decoder.feed(Mt3Vocab.SHIFT_BASE + 29)
        val end = decoder.feed(Mt3Vocab.pitchToken(60)).single() as DecodedNoteEvent.End
        assertEquals(0.29, end.timeSeconds, 1e-9)
    }

    @Test
    fun tiePreludeCarriesOpenNoteIntoNextChunk() {
        val decoder = Mt3StreamDecoder()
        decoder.beginChunk(0.0, 5.0)
        decoder.feed(Mt3Vocab.TIE)
        decoder.feed(Mt3Vocab.programToken(33))
        decoder.feed(Mt3Vocab.VELOCITY_BASE + 1)
        decoder.feed(Mt3Vocab.SHIFT_BASE + 490)
        decoder.feed(Mt3Vocab.pitchToken(38))

        decoder.beginChunk(5.0, 10.0)
        val forced = decoder.tieSectionTokenIds().toList()

        assertEquals(
            listOf(Mt3Vocab.programToken(33), Mt3Vocab.pitchToken(38), Mt3Vocab.TIE),
            forced,
        )

        // Feeding that forced prelude should keep the note open rather than close it at 5.0 s.
        val emitted = forced.flatMap { decoder.feed(it) }
        assertTrue(emitted.isEmpty())
        assertEquals(listOf(NoteKey(33, 38)), decoder.openKeys())
    }
}
