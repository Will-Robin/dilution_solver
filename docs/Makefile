DIL_SOLV=../src/dilution_solver
ROOT=../src/dilution_solver/*.py
DOE=../src/dilution_solver/doe

PACKAGE=$(ROOT) $(ROUTINES) $(DOE)

all: index.html

index.html: $(PACKAGE)
	pdoc ../src/dilution_solver --docformat "numpy" -o ./

clean:
	rm *.html
	rm dilution_solver/*.html
	rm search.js
