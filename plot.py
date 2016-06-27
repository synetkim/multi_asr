from pylab import *

#t = arange(0.0, 2.0, 0.01)
#s = sin(2*pi*t)
#plot(t, s)

legendlist=[]

def draw(fname):
	epochs=[]
	errs=[]
	v_epochs=[]
	v_errs=[]
	for line in open(fname,'r').readlines():
		line = line.replace("(%)","")
		if not "> epoch" in line: continue
		#if "train" in line:
		#	epoch = line.split(',')[0].split('epoch')[1]
		#	err = line.split('error')[-1]
		#	epochs.append(epoch)
		#	errs.append(err)
		if "valid" in line:
			epoch = line.split(',')[0].split('epoch')[1]
			err = line.split('error')[-1]
			v_epochs.append(epoch)
			v_errs.append(err)

	#plot(epochs, errs)
	#legendlist.append(fname.split("exp_pdnn")[-1])
	plot(v_epochs, v_errs)
	legendlist.append(fname.split("exp_pdnn")[-1]+"v")

#draw('mend.1/exp_pdnn.6ch/log/attendlstm.fine.log')
#draw('mend.2/exp_pdnn.6ch/log/attendlstm.fine.log')
draw('mend.15.2/exp_pdnn.6ch/log/attendlstm.fine.log')
draw('mend.15.3/exp_pdnn.6ch/log/attendlstm.fine.log')
draw('mend.15.4/exp_pdnn.6ch/log/attendlstm.fine.log')
draw('mend.15.5/exp_pdnn.6ch/log/attendlstm.fine.log')
#draw('attendlstm.fine.log3')
#draw('attendlstm.fine.log')
draw('/data-local1/sykim/mend.classification/exp_pdnn.close/log/dnn.fine.log')
draw('/data-local1/sykim/mend.classification/exp_pdnn.beamform/log/dnn.fine.log')
draw('/data-local1/sykim/mend.classification/exp_pdnn.noisy/log/dnn.fine.log')
draw('/data-local1/sykim/mend.classification/exp_pdnn.6ch/log/dnn.fine.log')

legend(legendlist,fontsize = 'x-small')

xlabel('epoch')
ylabel('errs')
title('learning curve')
grid(True)
savefig('test.png')
show()

