package dev.sakus.muscriptoreasy.inference

/** Human-readable names for MuScriptor's MT3_FULL_PLUS representative programs. */
object Mt3Instruments {
    const val DRUM_PROGRAM = 128

    private val representativeNames = mapOf(
        0 to "acoustic_piano",
        2 to "electric_piano",
        8 to "chromatic_percussion",
        16 to "organ",
        24 to "acoustic_guitar",
        26 to "clean_electric_guitar",
        29 to "distorted_electric_guitar",
        32 to "acoustic_bass",
        33 to "electric_bass",
        40 to "violin",
        41 to "viola",
        42 to "cello",
        43 to "contrabass",
        46 to "orchestral_harp",
        47 to "timpani",
        48 to "string_ensemble",
        50 to "synth_strings",
        52 to "voice",
        55 to "orchestra_hit",
        56 to "trumpet",
        57 to "trombone",
        58 to "tuba",
        60 to "french_horn",
        61 to "brass_section",
        64 to "soprano_and_alto_sax",
        66 to "tenor_sax",
        67 to "baritone_sax",
        68 to "oboe",
        69 to "english_horn",
        70 to "bassoon",
        71 to "clarinet",
        72 to "flutes",
        80 to "synth_lead",
        88 to "synth_pad",
        DRUM_PROGRAM to "drums",
    )

    fun nameForProgram(program: Int): String =
        representativeNames[program] ?: "program_$program"

    fun programsIn(events: List<DecodedNoteEvent>): List<Int> =
        events.asSequence()
            .filterIsInstance<DecodedNoteEvent.Start>()
            .map { if (it.isDrum) DRUM_PROGRAM else it.program }
            .distinct()
            .toList()
}
