%.xdmf: %.py
	python ${PYFLAGS} $<
%.0000.png: %.xdmf
	paraview $<
all: st07_navier_stokes_channel.xdmf
%.gif: st07_navier_stokes_channel.0000.png
	convert st07_navier_stokes_channel.00[01][0-9].png -loop 0 -delay 20 $@
clean:
	${RM} *.xdmf *.h5 *.png *.gif
	${RM} *~
.PHONY: all clean
