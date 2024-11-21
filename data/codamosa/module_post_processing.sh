#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

SMOKE_TEST_ONE=$SCRIPT_DIR/../smoke-test-results/exp-220608-1749/mosa
SMOKE_TEST_TWO=$SCRIPT_DIR/../smoke-test-results/exp-220610-1658/mosa
SMOKE_TEST_THREE=$SCRIPT_DIR/../smoke-test-results/exp-220610-1849/mosa

pushd $SMOKE_TEST_ONE > /dev/null
# Starting number of benchmarks in main project
#ls  | grep -v "^docs\." | grep -v "^benchmarks\." | grep -v "^scripts\."  | sed 's/-[01]//g' | sort | uniq | wc -l
echo "Total number of benchmarks in main project:"
ls  | sed 's/-[01]//g' | sort | uniq | wc -l
# Benchmarks on which we were able to run
echo "Benchmarks on which we are able to run:"
ls */statistics.csv | sed 's/-[01]\/statistics.csv//g' | sort | uniq | wc -l
echo "Over the following number of  modules:"
ls */statistics.csv | sed 's/-[01]\/statistics.csv//g' | sort | uniq | sed 's/\([^.]*\)\..*/\1/g' | sort |uniq | wc -l
# Benchmarks were we don't reach 1.0 coverage after a minute 
#tail -n 1 */statistics.csv | awk -F, '{print $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/ ==> \(.*\)-[01]\/statistics.csv <==/\1/g'  | uniq 
# Modules with 1 coverage at end
echo "Benchmarks which have 1 coverage at end:"
tail -n 1 */statistics.csv | awk -F, '{print $6, $NF}' | grep -B 1 '"1\..*"'  | grep ==  | sed 's/ ==> \(.*\)-[01]\/.*/\1/g'  | sort | uniq | wc -l
# Modules with 1 coverage at start
echo "Benchmarks which have 1 coverage at start:"
tail -n 1 */statistics.csv | awk -F, '{print $6, $NF}' | grep -B 1 '"1\..*" "1\..*"'  | grep ==  | sed 's/ ==> \(.*\)-[01]\/.*/\1/g'  | sort | uniq | wc -l
# Modules with < 1 coverage at end
echo "Benchmarks which have <1 coverage at end for at least one run"
tail -n 1 */statistics.csv | awk -F, '{print $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/==> \(.*\)-[01]\/.*/\1/g'  | sort | uniq | wc -l
# Take at most 20 modules for modules sharing a parent module
echo "Take at most 20 from each set sharing the same parent module"
tail -n 1 */statistics.csv | awk -F, '{print $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/==> \(.*\)-[01]\/.*/\1/g'  | sort | uniq | sed 's/^\(.*\)\..*/\1/g' | sort | uniq  -c  | awk 'START{sum=0} {sum+= 20 < $1 ? 20: $1} END {print sum}'

MORE_20_MODULES=$(tail -n 1 */statistics.csv | awk -F, '{print $6, $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/ ==> \(.*\)-[01]\/.*/\1/g'  | uniq | sed 's/^\(.*\)\..*/\1/g' | sort |uniq  -c | awk '{if ($1 > 20) print $2}')
DONT_MATCH=$(echo $MORE_20_MODULES |tr " " "|" )

rm -f /tmp/good_modules.txt
tail -n 1 */statistics.csv | awk -F, '{print $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/==> \(.*\)-[01]\/.*/\1/g'  | uniq | egrep -v $DONT_MATCH >> /tmp/good_modules.txt
# Offending modules
echo "$MORE_20_MODULES" |
        while read module_parent; do
                tail -n 1 */statistics.csv | awk -F, '{print $NF}' | grep -B 1 '"0\..*"'  | grep ==  | sed 's/==> \(.*\)-[01]\/.*/\1/g'  | grep $module_parent | grep -v porn | grep -v jizz | uniq | shuf -n 20 --random-source=/home/codamosa/scripts/seed.txt # Thanks, youtube_dl...
done  >> /tmp/good_modules.txt

rm -f ~/test-apps/good_modules_tmp.csv
while read module; do
        module_expr="${module//./\\.}"
        grep "$module_expr\$" ~/test-apps/all_potential_modules.csv >> ~/test-apps/good_modules_tmp.csv
done < /tmp/good_modules.txt

popd > /dev/null

pushd $SMOKE_TEST_TWO > /dev/null
rm -f /tmp/bad_modules_tmp.txt
for d in *; do
        test -e $d/statistics.csv; echo $d, $?;
done | grep -v ', 0' | sed 's/\([^-]*\)-[01].*/\1/' | sort | uniq >> /tmp/bad_modules_tmp.txt

sort ../time_out_commands.txt  | uniq | grep module-name | sed 's/.*--module-name \([^ ]*\) .*/\1/g'  | sort | uniq | while read module; do grep $module ~/test-apps/all_potential_modules.csv > ~/test-apps/questionable_modules.csv; done
popd > /dev/null

pushd $SMOKE_TEST_THREE > /dev/null
rm -f /tmp/bad_modules.txt
for d in *; do
        test -e $d/statistics.csv; echo $d, $?;
done | grep -v ', 0' | sed 's/\([^-]*\)-[01].*/\1/' | sort | uniq >> /tmp/bad_modules.txt

sort ../time_out_commands.txt  | grep module-name | sed 's/.*--module-name \([^ ]*\) .*/\1/g'  | sort | uniq >> /tmp/bad_modules.txt

BAD_MODULES=$(cat /tmp/bad_modules_tmp.txt /tmp/bad_modules.txt | sort | uniq )
DONT_MATCH=$(echo $BAD_MODULES |sed  's/ /$|,/g' | sed 's/\./\\\./g')
echo "Number of those modules which have to be timed out aggressively" 
egrep $DONT_MATCH ~/test-apps/good_modules_tmp.csv | sort | uniq | wc -l
echo "A module we will find later doesn't consistently produce a statistics.csv file, one on which MOSA 'fails to run' (one of the two runs in the smoke test fail)"
DONT_MATCH="${DONT_MATCH}\$|,ansible\\.modules\\.unarchive\$"
echo 1
echo "Final number of modules:"
egrep -v $DONT_MATCH ~/test-apps/good_modules_tmp.csv | sort | uniq | wc -l
egrep -v $DONT_MATCH ~/test-apps/good_modules_tmp.csv | sort | uniq > ~/test-apps/good_modules.csv
rm ~/test-apps/good_modules_tmp.csv
popd > /dev/null