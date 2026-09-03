package dev.sakus.muscriptoreasy.midi

import dev.sakus.muscriptoreasy.inference.DecodedNoteEvent
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

/**
 * Pure-Kotlin Standard MIDI File writer matching MuScriptor's fallback MIDI path.
 *
 * - SMF type 1
 * - 480 ticks/beat
 * - 120 BPM when no beat grid is available
 * - one track per MIDI program
 * - drums on channel 9
 * - melodic programs claim channels 0..8,10..15 in first-appearance order
 * - velocity 100
 */
object MidiExporter {
    const val DEFAULT_PPQ = 480
    const val DEFAULT_BPM = 120
    const val DEFAULT_VELOCITY = 100
    private const val DRUM_PROGRAM = 128

    fun encode(
        events: List<DecodedNoteEvent>,
        bpm: Int = DEFAULT_BPM,
        ppq: Int = DEFAULT_PPQ,
        velocity: Int = DEFAULT_VELOCITY,
    ): ByteArray {
        require(bpm > 0)
        require(ppq in 1..0x7fff)
        require(velocity in 1..127)

        val tempoMicros = (60_000_000.0 / bpm).roundToInt()
        val ordered = events.sortedWith(
            compareBy<DecodedNoteEvent>(
                { timeOf(it) },
                { if (isDrum(it)) 1 else 0 },
                { programOf(it) },
                { if (it is DecodedNoteEvent.End) 0 else 1 },
                { pitchOf(it) },
            ),
        )

        val programs = linkedSetOf<Int>()
        ordered.forEach { programs += if (isDrum(it)) DRUM_PROGRAM else programOf(it) }

        val channelByProgram = linkedMapOf<Int, Int>()
        val melodicChannels = (0..8).toMutableList().apply { addAll(10..15) }
        programs.forEach { program ->
            channelByProgram[program] = if (program == DRUM_PROGRAM) {
                9
            } else if (melodicChannels.isNotEmpty()) {
                melodicChannels.removeAt(0)
            } else {
                15
            }
        }

        val tracks = ArrayList<ByteArray>(programs.size + 1)
        tracks += metaTrack(tempoMicros)
        for (program in programs) {
            val programEvents = ordered.filter {
                (if (isDrum(it)) DRUM_PROGRAM else programOf(it)) == program
            }
            tracks += programTrack(
                program = program,
                channel = checkNotNull(channelByProgram[program]),
                events = programEvents,
                tempoMicros = tempoMicros,
                ppq = ppq,
                velocity = velocity,
            )
        }

        val out = ByteArrayOutputStream()
        out.writeAscii("MThd")
        out.writeInt32(6)
        out.writeInt16(1)
        out.writeInt16(tracks.size)
        out.writeInt16(ppq)
        tracks.forEach { data ->
            out.writeAscii("MTrk")
            out.writeInt32(data.size)
            out.write(data)
        }
        return out.toByteArray()
    }

    private fun metaTrack(tempoMicros: Int): ByteArray {
        val out = ByteArrayOutputStream()
        out.writeVlq(0)
        out.write(0xff)
        out.write(0x51)
        out.write(0x03)
        out.write((tempoMicros ushr 16) and 0xff)
        out.write((tempoMicros ushr 8) and 0xff)
        out.write(tempoMicros and 0xff)
        writeEndOfTrack(out)
        return out.toByteArray()
    }

    private fun programTrack(
        program: Int,
        channel: Int,
        events: List<DecodedNoteEvent>,
        tempoMicros: Int,
        ppq: Int,
        velocity: Int,
    ): ByteArray {
        val out = ByteArrayOutputStream()
        val name = if (program == DRUM_PROGRAM) "drums" else "program $program"

        writeMetaText(out, 0x03, name)

        // MuseScore may ignore a conductor-only tempo track, so upstream repeats
        // set_tempo on every note track too.
        out.writeVlq(0)
        out.write(0xff)
        out.write(0x51)
        out.write(0x03)
        out.write((tempoMicros ushr 16) and 0xff)
        out.write((tempoMicros ushr 8) and 0xff)
        out.write(tempoMicros and 0xff)

        out.writeVlq(0)
        out.write(0xc0 or channel)
        out.write(if (program == DRUM_PROGRAM) 0 else program.coerceIn(0, 127))

        var trackTick = 0
        events.forEach { event ->
            val absoluteTick = secondsToTicks(timeOf(event).coerceAtLeast(0.0), ppq, tempoMicros)
            val delta = (absoluteTick - trackTick).coerceAtLeast(0)
            trackTick = maxOf(trackTick, absoluteTick)
            out.writeVlq(delta)
            out.write((if (event is DecodedNoteEvent.Start) 0x90 else 0x80) or channel)
            out.write(pitchOf(event).coerceIn(0, 127))
            out.write(if (event is DecodedNoteEvent.Start) velocity else 0)
        }

        writeEndOfTrack(out)
        return out.toByteArray()
    }

    private fun secondsToTicks(seconds: Double, ppq: Int, tempoMicros: Int): Int =
        (seconds * ppq * 1_000_000.0 / tempoMicros).roundToInt().coerceAtLeast(0)

    private fun timeOf(event: DecodedNoteEvent): Double = when (event) {
        is DecodedNoteEvent.Start -> event.timeSeconds
        is DecodedNoteEvent.End -> event.timeSeconds
    }

    private fun programOf(event: DecodedNoteEvent): Int = when (event) {
        is DecodedNoteEvent.Start -> event.program
        is DecodedNoteEvent.End -> event.program
    }

    private fun pitchOf(event: DecodedNoteEvent): Int = when (event) {
        is DecodedNoteEvent.Start -> event.pitch
        is DecodedNoteEvent.End -> event.pitch
    }

    private fun isDrum(event: DecodedNoteEvent): Boolean = when (event) {
        is DecodedNoteEvent.Start -> event.isDrum
        is DecodedNoteEvent.End -> event.isDrum
    }

    private fun writeMetaText(out: ByteArrayOutputStream, type: Int, text: String) {
        val bytes = text.toByteArray(Charsets.UTF_8)
        out.writeVlq(0)
        out.write(0xff)
        out.write(type)
        out.writeVlq(bytes.size)
        out.write(bytes)
    }

    private fun writeEndOfTrack(out: ByteArrayOutputStream) {
        out.writeVlq(0)
        out.write(0xff)
        out.write(0x2f)
        out.write(0x00)
    }

    private fun ByteArrayOutputStream.writeAscii(value: String) {
        write(value.toByteArray(Charsets.US_ASCII))
    }

    private fun ByteArrayOutputStream.writeInt16(value: Int) {
        write((value ushr 8) and 0xff)
        write(value and 0xff)
    }

    private fun ByteArrayOutputStream.writeInt32(value: Int) {
        write((value ushr 24) and 0xff)
        write((value ushr 16) and 0xff)
        write((value ushr 8) and 0xff)
        write(value and 0xff)
    }

    private fun ByteArrayOutputStream.writeVlq(value: Int) {
        require(value >= 0)
        var v = value
        var buffer = v and 0x7f
        while (v ushr 7 != 0) {
            v = v ushr 7
            buffer = (buffer shl 8) or ((v and 0x7f) or 0x80)
        }
        while (true) {
            write(buffer and 0xff)
            if (buffer and 0x80 != 0) buffer = buffer ushr 8 else break
        }
    }
}
