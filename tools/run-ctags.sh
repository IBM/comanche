#!/bin/bash
ctags --languages=C++ --exclude=third_party --exclude=.git --exclude=build --exclude=out -R -f .tmp_tags & ctags --languages=C++ -a -R -f .tmp_tags src

#& ctags --languages=C++ -a -R -f .tmp_tags third_party\WebKit & move /Y .tmp_tags .tags
