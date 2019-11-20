%.xdmf: %.py
	python ${PYFLAGS} $<
all: st07_navier_stokes_channel.xdmf
clean:
	${RM} *.xdmf *.h5 *.png
	${RM} *~
.PHONY: all clean
