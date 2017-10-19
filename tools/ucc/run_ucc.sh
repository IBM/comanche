KIVATI_BASE=$HOME/kivati
SOURCE_FILE_PATTERN='*.cc *.c *.h *.hpp'

DIRS='src/lib/common '
DIRS+='src/include/common src/include/core'l

# auto collect include directories
#INCLUDE_DIRS=`cd $KIVATI_BASE/ ; find -P ./ -maxdepth 1 -type d -regex ".\/[a-zA-Z].*" `
#echo "Includes: $INCLUDE_DIRS"
#DIRS += $INCLUDE_DIRS

echo $DIRS > DIRS

for dir in $DIRS
do
    odir=`echo $dir | sed 's/\//_/g'`
    ./UCC -ascii -nolinks -dir $KIVATI_BASE/$dir $SOURCE_FILE_PATTERN  -outdir data_$odir 
done

echo "=== KEY METRICS SUMMARY ==="

# output physical-logical ratio
for dir in $DIRS
do
    odir=`echo $dir | sed 's/\//_/g'`
    ratio=`grep 'Ratio of Physical' data_$odir/C_CPP_outfile.txt`
    echo $ratio  $dir 
done

echo ""

# output cc1 ratio
for dir in $DIRS
do
    odir=`echo $dir | sed 's/\//_/g'`
    ratio=`grep 'Average_CC1' data_$odir/outfile_cyclomatic_cplx.txt`
    echo $ratio  $dir 
done

echo ""

# output line-comment ratios
for dir in $DIRS
do
    odir=`echo $dir | sed 's/\//_/g'`
    info=`grep 'CODE  Physical' data_$odir/C_CPP_outfile.txt`
    total_lines=`echo $info | awk '{ print $1 }'`
    blank_lines=`echo $info | awk '{ print $2 }'`
    whole_comments=`echo $info | awk '{ print $4 }'`
    embed_comments=`echo $info | awk '{ print $5 }'`
    comments=$(( whole_comments + embed_comments ))
    ratio=`echo $comments $total_lines | awk '{printf "%.2f \n", $1/$2 * 100}'`
    echo Total: $total_lines $'\t'Blank: $blank_lines $'\t' Comments: $comments $'\t' Comment Ratio: $ratio \% $dir
done
