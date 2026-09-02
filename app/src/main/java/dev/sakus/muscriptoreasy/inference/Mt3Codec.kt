package dev.sakus.muscriptoreasy.inference

/** Exact generated-token vocabulary used by MuScriptor's MT3 tokenizer. */
object Mt3Vocab {
    const val PAD = 0
    const val EOS = 1
    const val UNK = 2

    const val SHIFT_BASE = 3
    const val SHIFT_COUNT = 1001
    const val PITCH_BASE = SHIFT_BASE + SHIFT_COUNT // 1004
    const val PITCH_COUNT = 128
    const val VELOCITY_BASE = PITCH_BASE + PITCH_COUNT // 1132
    const val VELOCITY_COUNT = 2
    const val TIE = VELOCITY_BASE + VELOCITY_COUNT // 1134
    const val PROGRAM_BASE = TIE + 1 // 1135
    const val PROGRAM_COUNT = 130
    const val DRUM_BASE = PROGRAM_BASE + PROGRAM_COUNT // 1265
    const val DRUM_COUNT = 128
    const val GENERATED_VOCAB_SIZE = DRUM_BASE + DRUM_COUNT // 1393

    sealed interface Event {
        data class Shift(val ticks: Int) : Event
        data class Pitch(val pitch: Int) : Event
        data class Velocity(val value: Int) : Event
        data object Tie : Event
        data class Program(val program: Int) : Event
        data class Drum(val pitch: Int) : Event
        data class Special(val token: Int) : Event
    }

    fun decode(token: Int): Event {
        require(token in 0 until GENERATED_VOCAB_SIZE) { "Invalid MT3 token: $token" }
        return when {
            token < SHIFT_BASE -> Event.Special(token)
            token < PITCH_BASE -> Event.Shift(token - SHIFT_BASE)
            token < VELOCITY_BASE -> Event.Pitch(token - PITCH_BASE)
            token < TIE -> Event.Velocity(token - VELOCITY_BASE)
            token == TIE -> Event.Tie
            token < DRUM_BASE -> Event.Program(token - PROGRAM_BASE)
            else -> Event.Drum(token - DRUM_BASE)
        }
    }

    fun programToken(program: Int): Int {
        require(program in 0 until PROGRAM_COUNT)
        return PROGRAM_BASE + program
    }

    fun pitchToken(pitch: Int): Int {
        require(pitch in 0 until PITCH_COUNT)
        return PITCH_BASE + pitch
    }
}

data class NoteKey(val program: Int, val pitch: Int)

sealed interface DecodedNoteEvent {
    data class Start(
        val program: Int,
        val pitch: Int,
        val timeSeconds: Double,
        val isDrum: Boolean = false,
    ) : DecodedNoteEvent

    data class End(
        val program: Int,
        val pitch: Int,
        val timeSeconds: Double,
        val isDrum: Boolean = false,
    ) : DecodedNoteEvent
}

/**
 * Kotlin port of upstream MuScriptor's OpenNoteTracker.
 *
 * It consumes one generated token at a time and keeps the cross-chunk tie state
 * required for correct sustained notes and prelude forcing.
 */
class Mt3StreamDecoder(private val frameRate: Int = 100) {
    private val open = linkedMapOf<NoteKey, Double>()

    private var seekTime = 0.0
    private var nextSeekTime: Double? = null
    private var startTick = 0
    private var tickState = 0
    private var program: Int? = null
    private var velocity: Int? = null
    private var inPrologue = true
    private var skipRest = false
    private val tieSet = linkedSetOf<NoteKey>()
    private var chunkStarted = false

    fun beginChunk(seekTimeSeconds: Double, nextSeekTimeSeconds: Double?): List<DecodedNoteEvent> {
        val actions = mutableListOf<DecodedNoteEvent>()
        if (chunkStarted && inPrologue) {
            actions += endAll(this.seekTime)
        }

        seekTime = seekTimeSeconds
        nextSeekTime = nextSeekTimeSeconds
        startTick = kotlin.math.round(seekTimeSeconds * frameRate).toInt()
        tickState = startTick
        program = null
        velocity = null
        inPrologue = true
        skipRest = false
        tieSet.clear()
        chunkStarted = true
        return actions
    }

    fun feed(token: Int): List<DecodedNoteEvent> {
        val event = Mt3Vocab.decode(token)

        if (inPrologue) {
            when (event) {
                Mt3Vocab.Event.Tie -> {
                    inPrologue = false
                    velocity = null
                    val ended = open.keys.filter { it !in tieSet }
                    ended.forEach { open.remove(it) }
                    return ended.map { DecodedNoteEvent.End(it.program, it.pitch, seekTime) }
                }

                is Mt3Vocab.Event.Shift -> {
                    // Upstream treats a shift before the tie marker as a malformed chunk.
                    inPrologue = false
                    skipRest = true
                    return endAll(seekTime)
                }

                is Mt3Vocab.Event.Program -> program = event.program
                is Mt3Vocab.Event.Pitch -> program?.let { tieSet += NoteKey(it, event.pitch) }
                else -> Unit
            }
            return emptyList()
        }

        if (skipRest) return emptyList()

        return when (event) {
            is Mt3Vocab.Event.Shift -> {
                if (event.ticks > 0) tickState = startTick + event.ticks
                emptyList()
            }

            is Mt3Vocab.Event.Program -> {
                program = event.program
                emptyList()
            }

            is Mt3Vocab.Event.Velocity -> {
                velocity = event.value
                emptyList()
            }

            is Mt3Vocab.Event.Drum -> {
                val time = tickState.toDouble() / frameRate
                if (nextSeekTime == null || time < nextSeekTime!!) {
                    listOf(
                        DecodedNoteEvent.Start(128, event.pitch, time, isDrum = true),
                        DecodedNoteEvent.End(128, event.pitch, time + 0.01, isDrum = true),
                    )
                } else {
                    emptyList()
                }
            }

            is Mt3Vocab.Event.Pitch -> handlePitch(event.pitch)
            else -> emptyList()
        }
    }

    fun finish(): List<DecodedNoteEvent> {
        if (chunkStarted && inPrologue) return endAll(seekTime)
        val result = open.map { (key, onset) ->
            DecodedNoteEvent.End(key.program, key.pitch, onset + 0.01)
        }
        open.clear()
        return result
    }

    fun openKeys(): List<NoteKey> = open.keys.sortedWith(compareBy({ it.program }, { it.pitch }))

    /** Exact teacher-forced tie prelude for the next 5-second chunk. */
    fun tieSectionTokenIds(): IntArray {
        val tokens = ArrayList<Int>()
        var programState: Int? = null
        for (key in openKeys()) {
            if (key.program != programState) {
                tokens += Mt3Vocab.programToken(key.program)
                programState = key.program
            }
            tokens += Mt3Vocab.pitchToken(key.pitch)
        }
        tokens += Mt3Vocab.TIE
        return tokens.toIntArray()
    }

    private fun handlePitch(pitch: Int): List<DecodedNoteEvent> {
        val currentProgram = program ?: return emptyList()
        val currentVelocity = velocity ?: return emptyList()
        val time = tickState.toDouble() / frameRate
        if (nextSeekTime != null && time >= nextSeekTime!!) return emptyList()

        val key = NoteKey(currentProgram, pitch)
        val out = mutableListOf<DecodedNoteEvent>()
        if (open.remove(key) != null) {
            out += DecodedNoteEvent.End(currentProgram, pitch, time)
        }
        if (currentVelocity > 0) {
            open[key] = time
            out += DecodedNoteEvent.Start(currentProgram, pitch, time)
        }
        return out
    }

    private fun endAll(time: Double): List<DecodedNoteEvent> {
        val result = open.keys.map { DecodedNoteEvent.End(it.program, it.pitch, time) }
        open.clear()
        return result
    }
}
