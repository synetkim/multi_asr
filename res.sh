
in=$1

for x in $in/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
for x in $in/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
