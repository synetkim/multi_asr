
ln -s /data-local1/sykim/mend.baseline/data* .
cp /data-local1/sykim/mend.baseline/exp.multi ./ -rf
mkdir -p exp_pdnn.6ch.real
cp /data-local1/sykim/mend.baseline/exp_pdnn.6ch.multi/* ./exp_pdnn.6ch.multi/
ln -s /data-local1/sykim/mend.baseline/exp_pdnn.6ch.multi/data.multi ./exp_pdnn.6ch.multi/
touch ./exp_pdnn.6ch.multi/5.extra.fine.done
touch ./exp_pdnn.6ch.multi/2.extra.fine.done
touch ./exp_pdnn.6ch.multi/10.extra.fine.done
