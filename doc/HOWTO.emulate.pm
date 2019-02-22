To configure "simulated" persistent memory to the kernel, add (modified) boot option:

    memmap=X!Y where X is rsize, and Y is offset to allocate from.  For example:

    memmap=32G!480G


By default it will be fsdax.  To provision as devdax:

    sudo ndctl destroy-namespace all -f
    sudo ndctl create-namespace --mode devdax --map mem -e namespace0.0 -f
