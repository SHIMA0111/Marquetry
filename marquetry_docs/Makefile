# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

VERSION		  = 0.2.0

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

setup:
	@pip install sphinx==7.2.5 sphinx_rtd_theme sphinx-intl sphinx_design
	@mkdir -p $BUILDDIR _static _templates

setup-pdf: setup
	@echo "Installing LaTeX dependencies for PDF generation..."
	@pip install -U sphinx-rtd-theme sphinx_design
	@echo "Note: You need to have LaTeX installed on your system to build PDFs."
	@echo "For Ubuntu/Debian: apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-xetex fonts-ipaexfont"
	@echo "For macOS: Install MacTeX from http://www.tug.org/mactex/ and ensure /Library/TeX/texbin is in your PATH"
	@echo "For Windows: Install MiKTeX from https://miktex.org/ and ensure its bin directory is in your PATH"
	@echo "The 'latexmk' command must be available in your PATH for PDF generation."
	@echo "For Japanese PDF support, you need IPAex fonts: https://moji.or.jp/ipafont/ipaex00401/"

cleanup:
	@rm -rf ./_build/*

build: cleanup
	@$(SPHINXBUILD) -b html . ./_build
	@make -e SPHINXOPTS='-D language="ja"' html
	@cp -r _build/html _build/ja
	@rm -rf _build/html

check-latexmk:
	@which latexmk > /dev/null 2>&1 || (echo "Error: 'latexmk' command not found. Please install it:"; \
		echo "  - For Ubuntu/Debian: apt-get install texlive-latex-recommended texlive-latex-extra"; \
		echo "  - For macOS: Install MacTeX from http://www.tug.org/mactex/ and ensure it's in your PATH"; \
		echo "  - For Windows: Install MiKTeX from https://miktex.org/ and ensure it's in your PATH"; \
		exit 1)

pdf: cleanup check-latexmk
	@$(SPHINXBUILD) -b latex . ./_build/latex
	@echo "Building PDF..."
	@cd ./_build/latex && xelatex Marquetry.tex && xelatex Marquetry.tex
	@echo "PDF built in ./_build/latex/Marquetry.pdf"

pdf-ja: cleanup check-latexmk
	@$(SPHINXBUILD) -b latex -D language="ja" . ./_build/latex-ja
	@echo "Building Japanese PDF..."
	@cd ./_build/latex-ja && xelatex Marquetry.tex && xelatex Marquetry.tex
	@echo "Japanese PDF built in ./_build/latex-ja/Marquetry.pdf"

poupdate:
	@make gettext
	@sphinx-intl update -p _build/gettext -l ja

upload:
	@aws s3 cp ./_build s3://marquetry-doc/$(VERSION)/ --recursive

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
