#!/bin/bash

# echo "preparing CivilCommentsFine dataset in $SLURM_TMPDIR"
# start=`date +%s`

# if [ ! -d $SLURM_TMPDIR/civilcomments ]; then
#     echo "copying civilcomments.tar to $SLURM_TMPDIR"
#     cp $project/dataset/civilcomments.tar $SLURM_TMPDIR
#     echo "extracting civilcomments.tar to $SLURM_TMPDIR"
#     tar -xf $SLURM_TMPDIR/civilcomments.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/civilcomments.tar
# else
#     echo "civilcomments dataset already exists in $SLURM_TMPDIR, skipping extraction"
# fi

# end=`date +%s`
# echo "civilcomments dataset prepared in $((end - start)) seconds"

echo "preparing MultiNLI dataset in $SLURM_TMPDIR"
start=`date +%s`

if [ ! -d $SLURM_TMPDIR/multinli ]; then
    echo "copying multinli.tar to $SLURM_TMPDIR"
    cp $project/dataset/multinli.tar $SLURM_TMPDIR
    echo "extracting multinli.tar to $SLURM_TMPDIR"
    tar -xf $SLURM_TMPDIR/multinli.tar -C $SLURM_TMPDIR && rm -f $SLURM_TMPDIR/multinli.tar
else
    echo "multinli dataset already exists in $SLURM_TMPDIR, skipping extraction"
fi

end=`date +%s`
echo "multinli dataset prepared in $((end - start)) seconds"