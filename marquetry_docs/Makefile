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
	@pip install sphinx==7.2.5 sphinx_rtd_theme
	@mkdir -p $BUILDDIR _static _tmplates

cleanup:
	@rm -rf ./_build/*

build: cleanup
	@$(SPHINXBUILD) -b html . ./_build
	@make -e SPHINXOPTS='-D language="ja"' html
	@cp -r _build/html _build/ja
	@rm -rf _build/html

poupdate:
	@make gettext
	@sphinx-intl update -p _build/gettext -l ja

upload:
	@aws s3 cp ./_build s3://marquetry-doc/$(VERSION)/ --recursive

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
