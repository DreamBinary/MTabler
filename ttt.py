s = ['\\begin{tabular}{|l|r|r|r|r|}\n\\hline \\begin{tabular}{l} \nRelaxation \\\\\nconstraints\n\\end{tabular} & \\begin{tabular}{r} \nAvg. \\\\\nsolving \\\\\ntime (s)\n\\end{tabular} & \\begin{tabular}{r} \nMax. \\\\\nsolving \\\\\ntime (s)\n\\end{tabular} & \\multicolumn{2}{|c|}{\\begin{tabular}{r} \nSolved instances (in \\% of total \\\\\nnumber of instances)\n\\end{tabular}} \\\\\n\\hline \\hline \\(\\rho_{1}\\) & 12 & 80 & 65 & 35 \\\\\n\\hline \\(\\rho_{2}\\) & 46 & 250 & 75 & 25 \\\\\n\\hline\n\\end{tabular}']
s = s[0]
# extract all tables
import re
tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', s, re.DOTALL)
for table in tables:
    print(table)
    print("----")