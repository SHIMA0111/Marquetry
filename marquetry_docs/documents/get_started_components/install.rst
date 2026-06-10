==================================
Install Marquetry to your machine
==================================

Requirements
-------------
 - Python: 3.10 or later
 - Dependencies:
     - `NumPy <https://numpy.org>`_ >= 2.0 : **Python package** Scientific mathmatics library.
     - `Pandas <https://pandas.pydata.org>`_ >= 2.2 : **Python package** Data analysis library.
     - `pillow <https://pillow.readthedocs.io/en/stable/>`_ >= 10.4 : **Python package** Python image library.
     - `SciPy <https://scipy.org>`_ >= 1.13 : **Python package** Scientific computing library.

Options
--------
 - `Graphviz <https://graphviz.org>`_ :
    To generate compute graph and convert it to image (:meth:`marquetry.Model.plot`).
 - Python Packages:
     - `onnx <https://onnx.ai>`_ :
        Needed for the ONNX export (:meth:`marquetry.Model.export_onnx`).
        You can install it together with Marquetry by ``pip install "marquetry[onnx]"``.
        `onnxruntime <https://onnxruntime.ai>`_ is also useful to run the exported models.
     - `CuPy <https://cupy.dev>`_ :
        CuPy is NumPy compatible scientific mathmatics library working on CUDA GPU.
           - CuPy requires a machine installing CUDA GPU, if your machine doesn't have CUDA you can't install CuPy.
     - `Matplotlib <https://matplotlib.org/>`_:
        Matplotlib is a visualize module, this must not be needed but it is used in get started and examples. If you use the get started and examples, please install it.

Install
------------------
 1. (Option) Install Graphviz in your computer (needed only for :meth:`marquetry.Model.plot`)

    .. dropdown:: Mac (Homebrew)

       .. code-block:: shell

          brew install graphviz

    .. dropdown:: RedHat Linux

       .. code-block:: shell

          sudo yum install graphviz

    .. dropdown:: Windows

       For 64-bit: `graphviz-8.0.5 (64-bit) EXE installer <https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/8.0.5/windows_10_cmake_Release_graphviz-install-8.0.5-win64.exe>`_

       For 32-bit: `graphviz-8.0.5 (32-bit) EXE installer <https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/8.0.5/windows_10_cmake_Release_graphviz-install-8.0.5-win32.exe>`_

       (These links are from `Graphviz <https://graphviz.org>`_)

    .. dropdown:: Others

       Please check `the official download page <https://graphviz.org/download/>`_

 2. Install Marquetry using pip.

 .. code-block::

    pip install marquetry

 3. (Option) Install CuPy if your machine installed CUDA GPU.

    If you want to use CuPy, you need to install CUDA tools listed in
    https://docs.cupy.dev/en/stable/install.html#additional-cuda-libraries.

    CuPy installation should refer `CuPy Installation <https://docs.cupy.dev/en/stable/install.html>`_.

 4. (Option) Install Matplotlib

    .. code-block::

       pip install matplotlib

Verification
-------------
 - You can check if `Marquetry <../../index.html>`_ has been installed correctly by the following simple test.

 .. code-block:: python

    import marquetry as mq
    x = mq.random((3, 4, 5))
    print(x.shape)
    print(x)

 This output is

 .. code-block::

    (3, 4, 5)

    container([[[ 0.02946823 -2.72470816  0.50294601  1.2309693  -0.59347865]
                [-0.26269576  1.13579788  0.2002236  -0.3345718  -0.2201855 ]
                [ 0.50224944 -0.39815959  2.16678313 -0.05142171 -0.13123544]
                [ 1.7742589  -0.87390543  0.74750223 -0.10536388  0.0890647 ]]

               [[ 0.2746127   2.63377282  0.90952514 -0.12678728 -1.41712698]
                [-1.81469174 -0.12338727  0.25949144 -0.35687087  0.78317399]
                [ 0.44458767  0.47758409  2.55519755  0.91309785 -0.26906791]
                [ 0.33607339  0.05191208  0.80465005 -0.08434422 -1.66371255]]

               [[ 0.98159945 -0.78715625 -0.54765664  1.09341141 -0.48239709]
                [ 0.17202879  0.16912728  0.2007077   1.90741574 -0.19461772]
                [ 0.84163249  1.36121056 -0.41767145 -0.7239824  -1.18665633]
                [ 0.20006696 -0.03990122  0.77495972  0.23258396  0.65214153]]])
