import sys
import matplotlib.pyplot as plt
import random
from statistics import median
import scipy
import scipy.stats as sts
import numpy as np
import pandas as pd
from ete3 import Tree
import argparse
from scipy.optimize import curve_fit
import seaborn as sns

parser = argparse.ArgumentParser(description="This script generates several plots and analyses")
parser.add_argument("-o", "--orthofinder_dir", required=True, help="Directory containing the output of an OrthoFinder run")
parser.add_argument("-e", "--entry_list", required=True, help="File containing the list of entries to consider for computing the pangenome")
parser.add_argument("-n", "--name", default="", help="A string for the title of the plot")

args = parser.parse_args()

orthofinder_dir = args.orthofinder_dir
if orthofinder_dir[-1] == "/":
	orthofinder_dir = orthofinder_dir[:-1]
orthofile = orthofinder_dir + "/Orthogroups/Orthogroups.GeneCount.tsv"

sppfile, spplist = args.entry_list, []
spppos = {}

t = Tree(orthofinder_dir+"/Species_Tree/SpeciesTree_rooted_node_labels.txt", format=1)

for i in open(sppfile):
	spplist.append(i[:-1])

setdict = {}
for line in open(orthofile):
	chunk = line[:-1].split("\t")
	if chunk[0] == "Orthogroup":
		for i in range(len(chunk)):
			if chunk[i] in spplist:
				spppos[i] = chunk[i] 
				setdict[chunk[i]] = set()
	else:
		for i in range(len(chunk)):
			if i in spppos and int(chunk[i]) > 0:
				setdict[spppos[i]].add(chunk[0])

taxalist = []
for node in t.get_leaves():
	taxalist.append(node.name)

taxalist.sort()

coreset = set()
for i in spplist:
	if len(coreset) == 0: coreset = setdict[i]
	else:
		coreset = coreset.intersection(setdict[i])
corelen = len(coreset)

def report_stats(cat, x, y):
	for i in range(1, len(x)):
		print(cat[i], ":")
		print("Mean dist:", np.mean(x[i]))
		print("Median dist:", np.median(x[i]))
		print("Std dev dist:", np.std(x[i]))
		print("Mean overlap:", np.mean(y[i]))
		print("Median overlap:", np.median(y[i]))
		print("Std dev overlap:", np.std(y[i]))

def fig_data(cladelist, namelist):
	distlist= []
	intersectlist = []
	donelist, restdistlist, restintersectlist = [], [], []
	markerlist = ["o", "v", "^", ">", "<", "s", "D", "p", "P", "*"]
	colorlist = ["black", "salmon", "forestgreen", "dodgerblue", "mediumorchid", "lightcoral", "goldenrod", "deeppink", "silver", "lightsteelblue"]
	for clade in cladelist:
		distlist.append([])
		intersectlist.append([])
	for x in taxalist:
		if x not in spplist: continue
		for y in taxalist:
			if y not in spplist: continue
			elif x == y: continue
			elif x+y in donelist or y+x in donelist: continue
			else:
				donelist.append(x+y)
				for c in range(0,len(cladelist)):
					if x.split("_")[1][:6] in cladelist[c] and y.split("_")[1][:6] in cladelist[c]:
						distlist[c].append(t.get_distance(x,y))
						intersectlist[c].append(len(setdict[x].intersection(setdict[y]))-corelen)
						break
				else:
					restdistlist.append(t.get_distance(x,y))
					restintersectlist.append(len(setdict[x].intersection(setdict[y]))-corelen)
	x, y = [restdistlist], [restintersectlist]
	for a in distlist:
		x.append(a)
	for b in intersectlist:
		y.append(b)
	return(x, y, markerlist[0:len(x)], colorlist[0:len(y)])

fig = plt.figure(figsize=(200, 150))
gs = fig.add_gridspec(ncols=1, nrows=1)

ax1 = fig.add_subplot(gs[0,0])
f1_cladelist = [["TA1025"],["TA1010", "TA1007", "TA1009", "TA1023", "abieti"],["TA1031", "TA1032", "TA1026", "TA1055"]]
f1_cat = ["Other", "Boreal", "NorthAmericaB", "Mediterranean"]
f1_distlist, f1_intersectlist, f1_markerlist, f1_colorlist = fig_data(f1_cladelist, f1_cat)
f1_total_distlist, f1_total_intersectlist = [], []
print("Figure 4 statistics")
report_stats(f1_cat, f1_distlist, f1_intersectlist)

for i in range(len(f1_distlist)):
	plt.scatter(np.sqrt(f1_distlist[i]), f1_intersectlist[i], marker=f1_markerlist[i], color=f1_colorlist[i])
	f1_total_distlist = f1_total_distlist + f1_distlist[i]
	f1_total_intersectlist = f1_total_intersectlist + f1_intersectlist[i]
x, y = np.array(np.sqrt(f1_total_distlist)), np.array(f1_total_intersectlist)
m, b = np.polyfit(x, y, 1)
plt.xlabel("Square root of phylogenetic distance")
plt.ylabel("Number of shared PAV orthogroups")
plt.legend(labels=f1_cat)
plt.plot(x, x*m + b, color="black")

plt.show()


import statsmodels.api as sm
# Assuming x and y are your two variables
x = sm.add_constant(x)  # Add a constant term to the predictor variable
model = sm.OLS(y, x)
results = model.fit()

# Print the summary of the regression
print(results.summary())

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

lineagedict = {'Triab_TA100615M3':"NorthAmerica_A", 'Triab_TA10073M1':"NorthAmerica_B", 'Triab_TA100919M2':"NorthAmerica_B", 'Triab_TA10106M1':"NorthAmerica_B", 'Triab_TA10107M1':"NorthAmerica_B", 'Triab_TA10184M2':"NorthAmerica_C", 'Triab_TA10236M1':"NorthAmerica_B", 'Triab_TA102510M2':"Boreal", 'Triab_TA10252M2':"Boreal", 'Triab_TA10257M2':"Boreal", 'Triab_TA10258M1':"Boreal", 'Triab_TA10264M3':"Mediterranean", 'Triab_TA10277M3':"EastAsian", 'Triab_TA10311M2':"Mediterranean", 'Triab_TA10321M1':"Mediterranean", 'Triab_TA10555M1':"Mediterranean", 'Triab_L15831':"NorthAmerica_B", "Trichaptum_abietinum":"NorthAmerica_B"}

def symmetry(spplist, corelen, setdict):
	dfdict = {}
	dfdict_bonferroni = {}
	donelist = []
	pvallist = []
	accset = set()
	for i in setdict:
		accset = accset | setdict[i]
	accset = accset - coreset
	for A in spplist:
		dfdict[A] = []
		dfdict_bonferroni[A] = []
		for B in spplist:
			a, b = setdict[A], setdict[B]
			c11, c00, c10, c01 = len(a & b)-corelen, len(accset - a - b), len(a - b), len(b - a)
			matrix = [[c11, c10],[c01,c00]]
			print("Both:", c11)
			print("Only in ", A, ":", c10)
			print("Only in ", B, ":", c01)
			print("None:", c00)
			print("Total acc:", len(accset))
			print("Total core:", corelen)
			pv = mcnemar(matrix, exact=True).pvalue
			print(pv)
			dfdict[A].append(pv)
			if A != B and A+B not in donelist and B+A not in donelist:
				pvallist.append(pv)
	mptest, c = multipletests(pvallist, alpha=0.05, method='bonferroni')[0], 0
	donelist2 = []
	for A in spplist:
		for B in spplist:
			if A == B: dfdict_bonferroni[A].append(-1)
			elif A+B not in donelist and B+A not in donelist:
				if mptest[c] == True:
					dfdict_bonferroni[A].append(1)
				else:
					dfdict_bonferroni[A].append(0)
				c = c + 1
				donelist2.append(A+B)
			else: dfdict_bonferroni[A].append(-1)
	data, data2 = pd.DataFrame.from_dict(dfdict,orient="index", columns=spplist), pd.DataFrame.from_dict(dfdict_bonferroni, orient="index", columns=spplist)
	data.rename(columns={"Trichaptum_abietinum":"Triab_L15831"}, index={"Trichaptum_abietinum":"Triab_L15831"}, inplace=True)
	data2.rename(columns={"Trichaptum_abietinum":"Triab_L15831"}, index={"Trichaptum_abietinum":"Triab_L15831"}, inplace=True)
	return data, data2
import seaborn as sns
data, data2 = symmetry(spplist, corelen, setdict)

def define_d(t, spplist):
	taxalist = []
	for node in t.get_leaves():
		n = str(node)[3:]
		if n in spplist:
			print(n)
			taxalist.append(node.name)
			taxalist.sort()
	distmatrix = []
	for x in taxalist:
		distrow = []
		for y in taxalist:
			if x == y: distrow.append(0.0)
			else:
				distrow.append(t.get_distance(x,y))
		distmatrix.append(distrow)
	return(scipy.cluster.hierarchy.linkage(distmatrix), taxalist)

edited_t = Tree(str(t.write()).replace("Trichaptum_abietinum", "Triab_L15831"), format=1)

edited_spplist = []
for i in spplist:
	if i == "Trichaptum_abietinum":
		edited_spplist.append("Triab_L15831")
	else:
		edited_spplist.append(i)

d, tx = define_d(edited_t, edited_spplist)

lineage_color={"NorthAmerica_A":"limegreen", "NorthAmerica_B":"forestgreen", "NorthAmerica_C":"darkolivegreen", "EastAsian":"gold", "Boreal":"salmon","Mediterranean":"dodgerblue"}

row_colors = [lineage_color[lineagedict[lin]] for lin in tx]
pal = sns.dark_palette("#69d", reverse=False, as_cmap=True)
sns.clustermap(data2.loc[tx, tx], cmap=pal, center=0, row_linkage=d, col_linkage=d, row_colors=row_colors, col_colors=row_colors, linewidths=.05)

plt.show()

def venn_data(spplist, setdict, lineagedict):
	for i in setdict:
		accset = set()
		accset = accset | setdict[i]
	accset = accset - coreset
	lineagesetdict = {"Boreal":accset, "NorthAmerica_B":accset, "Mediterranean":accset}
	for pos in range(len(spplist)):
		if spplist[pos] == "Trichaptum_abietinum":
			spplist[pos] = "Triab_L15831"
	for A in spplist:
		if A == "Triab_L15831": A = "Trichaptum_abietinum"
		if lineagedict[A] in lineagesetdict.keys():
			lineagesetdict[lineagedict[A]] = lineagesetdict[lineagedict[A]] & setdict[A]
	return(lineagesetdict["Boreal"], lineagesetdict["NorthAmerica_B"], lineagesetdict["Mediterranean"])
b, n, m = venn_data(spplist, setdict, lineagedict)

from matplotlib_venn import venn3, venn3_circles, venn3_unweighted
v = venn3_unweighted(subsets=(len((n - b)-m), len((b - n)-m), len((n & b)-m), len((m - b)-n), len((m & n)-b), len((m & b)-n), len((n & b) & m)), set_labels = ('North America B', 'Boreal', 'Mediterranean'))
v.get_patch_by_id('100').set_color('forestgreen')
v.get_patch_by_id('010').set_color('salmon')
v.get_patch_by_id('001').set_color('dodgerblue')
v.get_patch_by_id('110').set_color('khaki')
v.get_patch_by_id('011').set_color('plum')
v.get_patch_by_id('101').set_color('paleturquoise')
v.get_patch_by_id('111').set_color('gainsboro')
c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1))

plt.title("Distribution of lineage-core orthogroups")
 
plt.show()

### Figure 3 ###
setdict = {}
OGlist = set()
og_ndict={}

for line in open(orthofile):
	chunk = line[:-1].split("\t")
	if chunk[0] == "Orthogroup":
		for i in range(len(chunk)):
			if chunk[i] in spplist:
				spppos[i] = chunk[i]
				setdict[chunk[i]] = set()
			if chunk[i] == "Trichaptum_abietinum":
				spppos[i] = "Triab_L15831"
				setdict["Triab_L15831"] = set()
	else:
		og_nlist = []
		for i in range(len(chunk)):
			if i in spppos:
				if int(chunk[i]) > 0:
					og_nlist.append(int(chunk[i]))
					setdict[spppos[i]].add(chunk[0])		
		if len(og_nlist) > 0:
			og_ndict[chunk[0]] = og_nlist
			OGlist.add(chunk[0])

rand_pan, rand_core = [], []
for i in spplist:
	rand_pan.append([])
	rand_core.append([])

while len(rand_pan[0]) < 1000:
	names, pan, core = [], [] , []
	Aset, Bset = set(), set()
	random.shuffle(spplist)
	for spp in spplist:
		names.append(spp)
		if len(pan) == 0:
			pan.append(len(setdict[spp]))
			core.append(len(setdict[spp]))
			Aset, Bset = setdict[spp], setdict[spp]
		else:
			Aset, Bset = Aset | setdict[spp], Bset & setdict[spp]
			pan.append(len(Aset))
			core.append(len(Bset))
	for i in range(0, len(pan)):
		rand_pan[i].append(pan[i])
		rand_core[i].append(core[i])

med_pan, med_core = [], []
for i in rand_pan:
	med_pan.append(median(i))
for i in rand_core:
	med_core.append(median(i))

OG_frequency = []
OG_freq_dict = {}
rev_og_freq_dict = {}

for i in OGlist:
	c = 0
	for e in setdict:
		if i in setdict[e]: c = c + 1
	OG_frequency.append(c)
	OG_freq_dict[i] = c
	if c not in rev_og_freq_dict:
		rev_og_freq_dict[c] = 1
	else:
		rev_og_freq_dict[c] = rev_og_freq_dict[c]+1
	
	
def heaps_law(n, B, d):
	return B*(n**d)

n_data = np.array(range(1, len(med_pan)+1))
n_data2 = np.array(range(0, len(rev_og_freq_dict)))
p_data = np.array(med_pan)-med_core[-1]

initial_guess = [med_pan[0], 0.2]

params, covariance = curve_fit(heaps_law, n_data, p_data, p0=initial_guess)
B_opt, k_opt = params

from sklearn.metrics import r2_score

p_fit = heaps_law(n_data, B_opt, k_opt)
r_squared = r2_score(p_data, p_fit)

print(f"Optimal beta (average accessory orthogroups per genome): {B_opt:.4f}")
print(f"Optimal lambda (openness): {k_opt:.4f}")
print("R-squared goodness of fit: "+  str(r_squared))

mae = np.mean(np.abs(p_data - p_fit))
print(f"MAE: {mae:.4f}")
rmse = np.sqrt(np.mean((p_data - p_fit)**2))
print(f"RMSE: {rmse:.4f}")
mape = np.mean(np.abs((p_data - p_fit) / p_data)) * 100
print(f"MAPE: {mape:.2f}%")

### Fig 4 ###
fig = plt.figure(figsize=(18, 8))
fig.suptitle(args.name, fontsize=16)
gs = fig.add_gridspec(ncols=5, nrows=2)

#Rarefaction curve
ax1 = fig.add_subplot(gs[0,0:3])
p1 = plt.violinplot(rand_pan[1:], positions=range(2,len(spplist)+1))
p2 = plt.violinplot(rand_core)
p3 = plt.plot(range(1,len(spplist)+1), med_pan, "b-")
p4 = plt.plot(range(1,len(spplist)+1), med_core, "r-")
plt.axhline(y=med_pan[-1], color='#CCCCCC', linestyle='-.')
plt.axhline(y=med_core[-1], color='#CCCCCC', linestyle='-.')
plt.xrange=[1,len(spplist)]
plt.text(2, med_pan[-1]-20, str(int(med_pan[-1])), fontsize=12)
plt.text(2, med_core[-1]+20, str(int(med_core[-1])), fontsize=12)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)

plt.ylabel('Cumulative number of orthogroups', fontsize=12)
fig.set_size_inches(18.5, 10.5)

#Heaps law fit
ax2 = fig.add_subplot(gs[1,0:3])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = chr(946)+f": {round(B_opt, 4):.4f}" + "\n" + ""+chr(955)+f": {round(k_opt, 4):.4f}" + "\n" + "R$^2$: "+  str(round(r_squared, 4))
ax2.text(0.35, 0.05, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='baseline',  horizontalalignment='right', bbox=props)
plt.scatter(n_data, p_data, label='Empirical Data')

plt.plot(n_data, p_fit, color='red', label=f'Fitted')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.xlabel('Number of genomes (n)')
plt.ylabel('Average number of accessory orthogroups (p)')
plt.legend()
plt.title("Heap's Law Fit")
plt.grid(True)

#Number of orthogroups versus number of genomes
ax3 = fig.add_subplot(gs[0,3:5])
acc_sum = 0
for e in OG_frequency:
	if e < max(OG_frequency) and e != 0: acc_sum = acc_sum + 1
plt.xlabel('Number of genomes')
plt.ylabel('Number of Orthogroups')
ax3.set_xlim([1,18])
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
hist, edges = np.histogram(OG_frequency, bins=len(n_data2))
width = np.diff(edges) # edges is bins
plt.xlabel('Number of genomes')
plt.ylabel('Number of Orthogroups')
n_data3 = np.append(n_data2, 17.5)
hist2= np.append(hist, acc_sum)
plt.bar(n_data3+1, hist2, width=0.9, align="center", ec="k")
ax3.set_xlim([0,20])
ax3.set_ylim([0,max(hist2)+1000])
plt.xticks(n_data)
ax3.patches[-1].set_facecolor('crimson')
for i in ax3.patches[0:-2]:
	i.set_facecolor('lightcoral')
ax3.text(n_data3[-1]+0.3, acc_sum + 180, "Sum\nAcc")
ax3.text(n_data3[-2]+0.4, acc_sum + 250, "Core")

#Variation in genes per orthogroup
ax4 = fig.add_subplot(gs[1,3:5])
d = {"orthogroup":[], "min":[], "max":[], "diff":[], "coreacc":[], "nspps":[], "sco":[]}

for i in og_ndict:
	verdict = "Core"
	if len(og_ndict[i]) < 17:
		verdict = "Acc"
	d["orthogroup"].append(i) 
	d["min"].append(min(og_ndict[i]))
	d["max"].append(max(og_ndict[i]))
	d["diff"].append(max(og_ndict[i])-min(og_ndict[i]))
	d["coreacc"].append(verdict)
	d["nspps"].append(len(og_ndict[i]))
	if min(og_ndict[i]) == 1 and max(og_ndict[i]) == 1 and verdict == "Core":
		d["sco"].append(True)
	else:
		d["sco"].append(False)

minmax_df = pd.DataFrame(d)
minmax_df_a = minmax_df[minmax_df["nspps"] > 1]
minmax_df_b = minmax_df_a[minmax_df_a["sco"] == False]
sns.histplot(data=minmax_df_b, x="diff", hue="coreacc", palette=["royalblue","crimson"], fill=True, binwidth=1, element="step")

ax4.set_xlim([0,15])
plt.xlabel('Difference between min and max genes per orthogroup')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)

plt.show()
