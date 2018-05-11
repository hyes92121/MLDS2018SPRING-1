#!/bin/sh
wget "https://www.dropbox.com/s/t62phnueypr9ee7/clr_conversation.txt?dl=0" -O clr_conversation.txt
wget "https://www.dropbox.com/s/1yychtpl3su799f/epoch3_data1920000.pt?dl=0" -O epoch3_data1920000.pt
python3 mao_2-2/test.py $1 $2
exit 0