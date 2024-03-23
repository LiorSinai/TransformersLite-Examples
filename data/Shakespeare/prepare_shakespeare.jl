using Unicode

filepath = "datasets/project_gutenberg_shakespeare.txt"
out_path = "datasets/shakespeare_plays.txt"

# text = open(filepath) do file
#     read(file, String)
# end

sections_to_skip = [
    1:80,            # Introduction
    80:2_855,        # THE SONNETS
    2_858:2_933,     # ALL’S WELL THAT ENDS WELL - preamble
    7_808:7_820,     # ALL’S WELL THAT ENDS WELL - epilogue
    7_822:7_937,     # THE TRAGEDY OF ANTONY AND CLEOPATRA - preamble
    14_463:14_549,   # AS YOU LIKE IT - preamble
    18_901:18_962,   # THE COMEDY OF ERRORS - preamble
    22_102:22_189,   # THE TRAGEDY OF CORIOLANUS - preamble
    28_547:28_641,   # CYMBELINE - preamble
    34_432:34_511,   # THE TRAGEDY OF HAMLET, PRINCE OF DENMARK - preamble
    41_130:41_208,   # THE FIRST PART OF KING HENRY THE FOURTH - preamble
    45_937:46_034,   # THE SECOND PART OF KING HENRY THE FOURTH - preamble
    51_099:51_130,   # THE SECOND PART OF KING HENRY THE FOURTH - epilogue
    51_133:51_290,   # THE LIFE OF KING HENRY THE FIFTH - preamble
    56_079:56_189,   # THE FIRST PART OF HENRY THE SIXTH - preamble
    60_709:60_820,   # THE SECOND PART OF KING HENRY THE SIXTH - preamble
    65_891:65_991,   # THE THIRD PART OF KING HENRY THE SIXTH - preamble
    71_013:71_156,   # KING HENRY THE EIGHTH - preamble
    74_600:74_630,   # _The order of the coronation_ - list of actor instructions
    76_127:76_160,   # KING HENRY THE EIGHTH - epilogue
    76_152:76_232,   # THE LIFE AND DEATH OF KING JOHN - preamble
    80_252:80_336,   # THE TRAGEDY OF JULIUS CAESAR - preamble
    84_899:84_983,   # THE TRAGEDY OF KING LEAR - preamble
    91_008:91_076,   # LOVE’S LABOUR’S LOST - preamble
    96_016:96_110,   # THE TRAGEDY OF MACBETH - preamble
    100_166:100_248, # MEASURE FOR MEASURE - preamble
    105_050:105_129, # THE MERCHANT OF VENICE - preamble
    109_221:109_306, # THE MERRY WIVES OF WINDSOR - preamble
    114_044:114_126, # A MIDSUMMER NIGHT’S DREAM - preamble
    117_529:117_636, # MUCH ADO ABOUT NOTHING - preamble
    122_129:122_197, # THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE - preamble
    128_411:128_546, # PERICLES, PRINCE OF TYRE - preamble
    132_563:132_648, # THE LIFE AND DEATH OF KING RICHARD THE SECOND - preamble
    136_891:136_998, # KING RICHARD THE THIRD - preamble
    143_366:143_485, # THE TRAGEDY OF ROMEO AND JULIET - preamble
    148_633:148_817, # THE TAMING OF THE SHREW - preamble
    153_501:153_573, # THE TEMPEST - preamble
    157_337:157_428, # THE LIFE OF TIMON OF ATHENS - preamble
    161_758:161_842, # THE TRAGEDY OF TITUS ANDRONICUS - preamble
    165_883:166_043, # TROILUS AND CRESSIDA - preamble
    172_087:172_160, # TWELFTH NIGHT; OR, WHAT YOU WILL - preamble
    176_583:176_654, # THE TWO GENTLEMEN OF VERONA - preamble
    180_834:181_023, # THE TWO NOBLE KINSMEN - preamble
    186_313:186_339, # THE TWO NOBLE KINSMEN - epilogue
    186_345:186_420, # THE WINTER’S TALE - preamble
    191_356:191_742, # A LOVER’S COMPLAINT
    191_743:192_324, # THE PASSIONATE PILGRIM
    192_325:192_418, # THE PHOENIX AND THE TURTLE
    192_419:194605,  # THE RAPE OF LUCRECE
    194_606:196_040, # VENUS AND ADONIS
    196_041:196_391, # license
]

lines_to_skip = Set{Int}()
for section in sections_to_skip
    push!(lines_to_skip, section...)
end

lines = String[]
starting_words = [
    "SCENE",
    "ACT",
    "EPILOGUE",
    "FINIS",
    #"Enter",
    #"Re-enter",
    #" " # usually incidates an action
]
#actor_instruction = r"^\s*\[.+\]\s*$"
open(filepath) do file
    for (line_number, line) in enumerate(eachline(file))
        if !(line_number in lines_to_skip) &&
           !any([startswith(line, word) for word in starting_words]) #&&
           #isnothing(match(actor_instruction, line))
            push!(lines, line)
        end
    end
end
text = join(lines, "\n")
text = replace(text, r"\n\n+"=>"\n\n")
text = replace(text, r"\d+"=>"") # there are very few numbers in the text
#text = replace(text, actor_instruction=>"")
text = replace(text, "\t"=>" ")
text = replace(text, " & "=>" AND ")
text = replace(text, r"&c.?"=>"etc.")
text = replace(text, "'"=>"")
## Remove unicode
text = Unicode.normalize(text, stripmark=true)
text = replace(text, "…"=>"")
text = replace(text, "Æ"=>"Ae")
text = replace(text, "æ"=>"ae")
text = replace(text, "œ"=>"oe")

open(out_path, "w") do file
    write(file, text[2:end]) # remove first new line
end

characters = join(sort(collect(Set(text))))
