%.xdmf: %.py
	python ${PYFLAGS} $<
all: st08_navier_stokes_cylinder.xdmf
clean:
	${RM} *.xdmf *.h5 *.png *~
.PHONY: all clean
