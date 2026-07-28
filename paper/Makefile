# Makefile for LaTeX projects using BibLaTeX/Biber
# Usage: make [TARGET=paper-fr]
#        make clean

# Declare all targets as phony to force execution
.PHONY: all clean cleanall help

# Default target file (override with: make TARGET=paper-fr)
TARGET ?= paper

# Default target builds the TARGET pdf
all: $(TARGET)

# Rule to build a specific document
$(TARGET):
	@echo "Building $@.pdf from $@.tex..."
	xelatex -interaction=nonstopmode $@.tex
	xelatex -interaction=nonstopmode $@.tex
	biber $@
	xelatex -interaction=nonstopmode $@.tex
	xelatex -interaction=nonstopmode $@.tex
	@echo "Build of $@.pdf complete."

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.bcf *.blg *.run.xml *.fls *.fdb_latexmk *.synctex.gz *.dvi *.bak *-blx.bib *.spl
	@echo "Clean complete."

# Clean all including PDFs
cleanall: clean
	@echo "Removing PDF files..."
	rm -f *.pdf
	@echo "Clean complete."

# Help command
help:
	@echo "Makefile for LaTeX projects with BibLaTeX/Biber"
	@echo ""
	@echo "Available targets:"
	@echo "  make                    - Build paper.pdf (default)"
	@echo "  make TARGET=paper-fr    - Build paper-fr.pdf instead"
	@echo "  make clean              - Remove all temporary LaTeX files"
	@echo "  make cleanall           - Remove all temporary files and PDFs"
	@echo "  make help               - Display this help message"