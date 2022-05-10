
export PS1='\[\033[1;31m\]\u@\h \[\033[1;34m\]\W # \[\033[1;37m\]'

if [ -f /etc/bash_completion ]; then source /etc/bash_completion; fi
echo
echo "Welcome to this Docker container, type \"make help\" to get some help."
echo
