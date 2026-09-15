conda activate SparkWorkshop

conda env list

python -c "
import numpy
import pandas
import pyspark
import matplotlib

print('All packages imported successfully!')
print('PySpark version:', pyspark.__version__)
"

conda deactivate