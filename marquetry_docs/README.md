# Marquetry Documentation

This directory contains the documentation for the Marquetry project in reStructuredText (RST) format.

## Building Documentation

### Setup

To set up the documentation build environment:

```bash
make setup
```

This will install the necessary Python packages (Sphinx, sphinx_rtd_theme, sphinx-intl, sphinx_design) for building the documentation.

### Building HTML Documentation

To build the HTML documentation:

```bash
make build
```

This will generate HTML documentation in both English and Japanese in the `_build` directory.

## PDF Generation

### Setup for PDF Generation

To set up the environment for PDF generation:

```bash
make setup-pdf
```

This will install the necessary Python packages (Sphinx, sphinx_rtd_theme, sphinx-intl, sphinx_design) and provide instructions for installing LaTeX and required fonts.

### Building PDF Documentation

To build the PDF documentation in English:

```bash
make pdf
```

The generated PDF will be available at `_build/latex/Marquetry.pdf`.

To build the PDF documentation in Japanese:

```bash
make pdf-ja
```

The generated Japanese PDF will be available at `_build/latex-ja/Marquetry.pdf`.

Note: The PDF generation process uses XeLaTeX directly to ensure proper handling of Japanese text. This bypasses the standard LaTeX build process and ensures compatibility with the xeCJK package.

### Requirements for PDF Generation

To generate PDFs, you need:

1. LaTeX installed on your system:
   - Ubuntu/Debian: `apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-xetex fonts-ipaexfont`
   - macOS: Install [MacTeX](http://www.tug.org/mactex/)
   - Windows: Install [MiKTeX](https://miktex.org/)

2. The `latexmk` command must be available in your PATH:
   - This is included with most LaTeX distributions but may need to be installed separately
   - For macOS with MacTeX: Make sure `/Library/TeX/texbin` is in your PATH
   - For Windows with MiKTeX: Make sure MiKTeX's bin directory is in your PATH

3. XeLaTeX must be available (included in the LaTeX distributions mentioned above):
   - The build process uses XeLaTeX instead of pdfLaTeX for better CJK support
   - This is especially important for Japanese documentation

4. For Japanese PDF support, you need IPAex fonts:
   - Download from: https://moji.or.jp/ipafont/ipaex00401/
   - Install the fonts on your system

## Translation

To update translation files:

```bash
make poupdate
```

This will generate/update the translation files in the `locales` directory.
