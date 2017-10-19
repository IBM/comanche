################################################################################
# colors.gdb
# Rémi Attab (remi.attab@gmail.com), 5 Apr 2013
# FreeBSD-style copyright and disclaimer apply
#
# gdb script that colorizes the output of several gdb functions.
#
# It accomplishes this feat by adding a hook the select gdb functions and
# redirecting the output of the function into a temp file int the current
# folder. In the post hook for the same function, this file is read back and
# piped through ungodly sed commands which colorizes the output.
#
# It also colorizes the prompt as an extra bonus.
#
# Currently supported gdb functions include:
# - backtrace
# - up
# - down
# - frame
# - info threads
# - thread
#
# WARNING: This script introduces a number of annoying behaviours into gdb.
#
#    First up, hooked commands that fail (eg. calling backtrace when no program
#    running or calling down on frame 0) will screw up the output
#    of any following commands. This happens because the post hook is never
#    executed (no idea why) and so the logging remains enabled. To fix this,
#    call cleanup-color-pipe.
#
#    Next, this script uses the logging functionality in gdb so if you're using
#    it to record gdb outputs, this script will break your output whenever you
#    call one of the hook-ed commands.
#
################################################################################


#------------------------------------------------------------------------------#
# Utils
#------------------------------------------------------------------------------#

define setup-color-pipe
    shell rm -f ./.gdb-color-pipe
    set logging redirect on
    set logging on ./.gdb-color-pipe
end

define cleanup-color-pipe
    set logging off
    set logging redirect off
    shell rm -f ./.gdb-color-pipe
end

document cleanup-color-pipe
    Disables command redirection and removes the color pipe file.
    Syntax: cleanup-color-pipe
end


define do-generic-colors
    # 1. Function names and the class they belong to
    # 2. Function argument names
    # 3. Stack frame number
    # 4. Thread id colorization
    # 5. File path and line number
    shell cat ./.gdb-color-pipe | \
        sed -r "s_([^<])(\b([a-zA-Z0-9_]+::)?[a-zA-Z0-9_\.@]+)( ?)\(_\1$(tput setaf 6)$(tput bold)\2$(tput sgr0)\4(_g" | \
        sed -r "s_([a-zA-Z0-9_#]*)=_$(tput setaf 2)$(tput bold)\1$(tput sgr0)=_g" | \
        sed -r "s_^(#[0-9]*)_$(tput setaf 1)$(tput bold)\1$(tput sgr0)_" | \
        sed -r "s_^([ \*]) ([0-9]+)_$(tput bold)$(tput setaf 6)\1 $(tput setaf 1)\2$(tput sgr0)_" | \
        sed -r "s_(\.*[/A-Za-z0-9\+_\.\-]*):([0-9]+)\$_$(tput setaf 4)\1$(tput sgr0):$(tput setaf 3)\2$(tput sgr0)_"
end

#------------------------------------------------------------------------------#
# Prompt
#------------------------------------------------------------------------------#

# The \001 and \002 tell libreadline to ignore the color codes
set prompt \001\033[01;34m\002(gdb) \001\033[0m\002


#------------------------------------------------------------------------------#
# backtrace
#------------------------------------------------------------------------------#

define hook-backtrace
    setup-color-pipe
end

define hookpost-backtrace
    do-generic-colors
    cleanup-color-pipe
end


#------------------------------------------------------------------------------#
# up
#------------------------------------------------------------------------------#

define hook-up
    setup-color-pipe
end

define hookpost-up
    do-generic-colors
    cleanup-color-pipe
end


#------------------------------------------------------------------------------#
# down
#------------------------------------------------------------------------------#

define hook-down
    setup-color-pipe
end

define hookpost-down
    do-generic-colors
    cleanup-color-pipe
end


#------------------------------------------------------------------------------#
# frame
#------------------------------------------------------------------------------#

define hook-frame
    setup-color-pipe
end

define hookpost-frame
    do-generic-colors
    cleanup-color-pipe
end


#------------------------------------------------------------------------------#
# info threads
#------------------------------------------------------------------------------#

define info hook-threads
    setup-color-pipe
end

define info hookpost-threads
    do-generic-colors
    cleanup-color-pipe
end


#------------------------------------------------------------------------------#
# thread
#------------------------------------------------------------------------------#

define hook-thread
    setup-color-pipe
end

define hookpost-thread
    do-generic-colors
    cleanup-color-pipe
end
