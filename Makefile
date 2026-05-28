SHELL  := /bin/bash
TEX    = main-v6
LOGDIR = $(HOME)/tmp

.PHONY: pdf clean

pdf: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	@mkdir -p $(LOGDIR)
	tectonic --keep-logs -c minimal -Z shell-escape $(TEX).tex 2>&1 | grep -vE "Overfull|Underfull"; test $${PIPESTATUS[0]} -eq 0
	mv -f $(TEX).log $(LOGDIR)/$(TEX).log

clean:
	rm -f $(TEX).pdf $(LOGDIR)/$(TEX).log
