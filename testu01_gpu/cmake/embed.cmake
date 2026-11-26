# cmake -DIN=file -DOUT=file.inc -P embed.cmake: the file IN as a raw string literal, written to OUT
file(READ ${IN} text)
file(WRITE ${OUT}.tmp "R\"TESTU01_GPU(${text})TESTU01_GPU\"\n")
file(RENAME ${OUT}.tmp ${OUT})
