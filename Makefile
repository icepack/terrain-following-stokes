stokes-tfc.pdf: stokes-tfc.tex demo/testing/flowchart.pdf
	latexmk -pdf stokes-tfc.tex

demo/testing/flowchart.pdf: demo/testing/flowchart.gv
	dot -Tpdf $< > $@

