import pandas as pd
import itertools
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
import matplotlib.cm as cm
import numpy as np
import matplotlib.style as style
import matplotlib.patches as mpatches
from itertools import repeat
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

def foo2(l):
    yield from itertools.product(*([l] * 2))

def foo3(l):
    yield from itertools.product(*([l] * 3))

def foo6(l):
    yield from itertools.product(*([l] * 6))

def clean_db(X):
    # drop 0 variance columns:
    del_col = []
    for i in X.columns[:-1]:
        if np.std(X.loc[:, i]) < 0.01 * np.mean(X.loc[:, i]):
            del_col.append(i)
    print(del_col, 'have 1% variance, will be dropped from db')
    X = X.drop(del_col, axis=1)

    # drop 90% correlated columns
    corr_db = pd.DataFrame(np.corrcoef(X.transpose().astype(float)),
                           index=X.columns,
                           columns=X.columns)
    del_col = []
    for c_index, c in enumerate(corr_db.columns):
        for b in corr_db.index[c_index + 1:]:
            if corr_db.loc[c, b] >= 0.8 and b != c:
                # print(c, ' and ', b, ' are strongly associated: ', corr_db.loc[c, b])
                if b not in del_col:
                    del_col.append(b)
    print("deleting column ", del_col)
    print("Total deleted columns = ", len(del_col))
    X = X.drop(del_col, axis=1)
    return X

def FeatureEnrichment(data):

    relaventColumns = data.columns[12:] #Only Trios of codons

    #Single codon features

    dataA = data.copy()
    dataC = data.copy()
    dataG = data.copy()
    dataT = data.copy()

    for col in relaventColumns:
        if col.count('A') == 2:
            dataA[col+"2"] = data[col]
        if col.count('C') == 2:
            dataC[col+"2"] = data[col]
        if col.count('T') == 2:
            dataT[col+"2"] = data[col]
        if col.count('G') == 2:
            dataG[col+"2"] = data[col]

        if col.count('A') == 3:
            dataA[col+"2"] = data[col]
            dataA[col + "3"] = data[col]
        if col.count('C') == 3:
            dataC[col+"2"] = data[col]
            dataC[col + "3"] = data[col]
        if col.count('G') == 3:
            dataG[col+"2"] = data[col]
            dataG[col + "3"] = data[col]
        if col.count('T') == 3:
            dataT[col+"2"] = data[col]
            dataT[col + "3"] = data[col]

    relaventColumnsA = dataA.columns[12:]
    relaventColumnsC = dataC.columns[12:]
    relaventColumnsG = dataG.columns[12:]
    relaventColumnsT = dataT.columns[12:]

    data['A'] = dataA[[str for str in relaventColumnsA if any(sub in str for sub in ['A'])]].sum(axis=1)
    data['G'] = dataG[[str for str in relaventColumnsG if any(sub in str for sub in ['G'])]].sum(axis=1)
    data['C'] = dataC[[str for str in relaventColumnsC if any(sub in str for sub in ['C'])]].sum(axis=1)
    data['T'] = dataT[[str for str in relaventColumnsT if any(sub in str for sub in ['T'])]].sum(axis=1)
    data['%A'] = data['A'] / 3 / data['# Codons']
    data['%G'] = data['G'] / 3 / data['# Codons']
    data['%C'] = data['C'] / 3 / data['# Codons']
    data['%T'] = data['T'] / 3 / data['# Codons']

    #Paired codon data

    relaventColumns = data.columns[12:]

    data1 = data.copy()
    data1['TTT2'] = data['TTT']
    data1['GGG2'] = data['GGG']
    data1['AAA2'] = data['AAA']
    data1['CCC2'] = data['CCC']

    relaventColumns2 = data1.columns[12:]

    for x in foo2('TGCA'):
        data[x[0]+x[1]] = data1[[str for str in relaventColumns2 if any(sub in str for sub in [x[0]+x[1]])]].sum(axis=1)
        data['%' + x[0]+x[1]] = data[x[0]+x[1]] / 2 / data['# Codons']


    #precentage of codons in trios

    for x in foo3('TGCA'):
        #if(x[0]+x[1]+x[2] != 'GGG'): #No GGG
        data['%' + x[0]+x[1]+x[2]] = data[x[0]+x[1]+x[2]] / data['# Codons']

    cols2 = []
    for x in foo2('TGCA'):
        cols2.append('%' + x[0] + x[1])
        data['%of2'] = data[cols2].sum(axis=1)

    cols3 = []
    for x in foo3('TGCA'):
        #if (x[0] + x[1] + x[2] != 'GGG'):
        cols3.append('%' + x[0] + x[1] + x[2])
        data['%of3'] = data[cols3].sum(axis=1)

    data['%of1'] = data['%A'] + data['%C'] + data['%T'] + data['%G']

    print("average of 1: " + str(data['%of1'].mean()))
    print("average of 2: " + str(data['%of2'].mean()))
    print("average of 3: " + str(data['%of3'].mean()))

    #Entropy

    c1 = ['%A', '%G' ,'%C', '%T']
    data['entropy1'] = -1 * (data[c1] * np.log2(data[c1])).sum(axis = 1)

    c2 = ['%'+x[0]+x[1] for x in foo2('TGCA')]
    data['entropy2'] = -1 * (data[c2] * np.log2(data[c2])).sum(axis=1)

    c3 = ['%'+x[0]+x[1]+x[2] for x in foo3('TGCA') ]#if x != ('G','G','G')]
    data['entropy3'] = -1 * (data[c3] * np.log2(data[c3])).sum(axis=1)

    return data

def FeatureEnrichmentBi(data):

    relaventColumns = data.columns[9:]

    for x in foo6('tcga'):
        data['%' + x[0]+x[1]+x[2]+x[3]+x[4]+x[5]] = data[x[0]+x[1]+x[2]+x[3]+x[4]+x[5]] / data['# Codon Pairs']

    c6 = ['%' + x[0]+x[1]+x[2]+x[3]+x[4]+x[5] for x in foo6('tcga')]
    data['entropy2'] = -1 * (data[c6] * np.log2(data[c6])).sum(axis=1)
    return data

def Preprocess(data):
    data = data[data['Organelle'] == 'genomic']
    #data2 = data.copy()
    data.pop('Division')
    data.pop('Organelle')
    data.pop('Taxid')
    data.pop('Species')
    data.pop('Assembly')
    data.pop('Translation Table')
    data.pop('Unnamed: 0')
    #data = clean_db(data)
    return data

def Visualize(data):


    sns.violinplot(data=data, x="Organelle", y="entropy1")
    plt.title('Entropy 1 for each Organelle')
    plt.savefig("Entropy1.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="entropy2")
    plt.title('Entropy 2 for each Organelle')
    plt.savefig("Entropy2.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="entropy3")
    plt.title('Entropy 3 for each Organelle')
    plt.savefig("Entropy3.jpg")
    plt.show()

    #% of 1 codon visualization

    sns.violinplot(data=data, x="Organelle", y="%A")
    plt.title('%A for each Organelle')
    plt.savefig("A.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%C")
    plt.title('%C for each Organelle')
    plt.savefig("C.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%G")
    plt.title('%G for each Organelle')
    plt.savefig("G.jpg")
    plt.show()

    sns.violinplot(data=data, x="Organelle", y="%T")
    plt.title('%T for each Organelle')
    plt.savefig("T.jpg")
    plt.show()

    #% of single codonds violine plot

    #% codong plots

    data_to_plot = [data['%A'], data['%C'], data['%T'], data['%G']]
    red_patch = mpatches.Patch(color='blue')
    pos   = [1, 2, 3, 4]
    label = ['%A','%C','%T','%G']

    fake_handles = repeat(red_patch, len(pos))

    plt.figure()
    ax = plt.subplot(111)
    plt.violinplot(data_to_plot, pos, vert=False)
    ax.legend(fake_handles, label)
    plt.title("Codon % for genomic organelle")
    plt.show()

    #entropy plots

    data_to_plot = [data['entropy1'], data['entropy2'], data['entropy3']]
    red_patch = mpatches.Patch(color='blue')
    pos   = [1, 2, 3]
    label = ['entropy1','entropy2','entropy3']

    fake_handles = repeat(red_patch, len(pos))

    plt.figure()
    ax = plt.subplot(111)
    plt.violinplot(data_to_plot, pos, vert=False)
    ax.legend(fake_handles, label)
    plt.title("Entropy for genomic organelle")
    plt.show()

def Kmeans(X):

    distrotions = []
    silhouette_avg_n_clusters = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

    for k in range_n_clusters:

        print("Fitting Kmeans model with k = " + str(k))
        kmeanModel = KMeans(n_clusters=k)
        clusters = kmeanModel.fit_predict(X)
        distrotions.append(kmeanModel.inertia_)
        # silhouette_avg = silhouette_score(X, clusters)
        # print("For n_clusters =", k,
        #       "The average silhouette_score is :", silhouette_avg)

        #silhouette_avg_n_clusters.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        #sample_silhouette_values = silhouette_samples(X, clusters)

    plt.figure(figsize=(16,8))
    plt.plot(range_n_clusters[20:24], distrotions[20:24], 'bx-')
    plt.xlabel('K')
    plt.yscale('linear')
    plt.ylabel('Distortion')
    plt.title("Elbow graph for Kmean")
    plt.savefig("Kmeans-Elbow.jpg")
    plt.show()

    # plt.figure(figsize=(16, 8))
    # plt.plot(range_n_clusters, silhouette_avg_n_clusters, 'bx-')
    # plt.xlabel('K')
    # plt.ylabel('Silhouette score')
    # plt.title("Elbow graph for Kmean")
    # plt.savefig("Kmeans-Elbow-silhouette.jpg")
    # plt.show()

def DBSCAN(data):
    pass

def makeEntropyBins(data):

    x1 = data['entropy1']
    x2 = data['entropy2']
    x3 = data['entropy3']

    hist1, bin_edges = np.histogram(x1, bins=10)
    hist2, bin_edges = np.histogram(x2, bins=10)
    hist3, bin_edges = np.histogram(x3, bins=10)

    print(hist1)
    print("-----")
    print(hist2)
    print("-----")
    print(hist3)
    print("-----")


    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10)

    print(str(x1.mean()) + 'std' + str(np.std(x1)))
    print(str(x2.mean()) + 'std' + str(np.std(x2)))
    print(str(x3.mean()) + 'std' + str(np.std(x3)))

    plt.hist(x1, **kwargs, label= 'Entropy 1')
    plt.hist(x2, **kwargs, label= 'Entropy 2')
    plt.hist(x3, **kwargs, label= 'Entropy 3')
    plt.legend()
    plt.title('Entropy histogram - 10 bins')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    #plt.savefig("entropy - histograms.jpg")
    plt.show()

def makeCorrelationMatrix(data):

    #data = data[[col for col in data.columns if ((col[0] == '%' ) | (col[0] == 'e'))]]

    # data.pop("%of1")
    # data.pop("%of2")
    # data.pop("%of3")

    corr = data.corr()
    components = list()
    visited = set()
    for col in data.columns:
        if col in visited:
            continue

        component = set([col, ])
        just_visited = [col, ]
        visited.add(col)
        while just_visited:
            c = just_visited.pop(0)
            for idx, val in corr[c].items():
                if abs(val) > 0.0 and idx not in visited:
                    just_visited.append(idx)
                    visited.add(idx)
                    component.add(idx)
        components.append(component)

    for component in components:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr.loc[component, component], cmap="Reds")

    #
    # fig, ax = plt.subplots(figsize=(15, 15))  # Sample figsize in inches
    #
    # corr = data.corr()
    #
    # sns.heatmap(corr, cmap="Blues", annot=False, ax = ax, linewidths=5)
    if len(data.columns[0]) == 1:
        plt.title("Correlation for single codons")
    if len(data.columns[0]) == 2:
        plt.title("Correlation for paired codons")
    else:
        plt.title("Codon correlation")
    plt.show()

def findOptimalK(X):

    distrotions = []
    diffs = []

    kmeanModel = KMeans(n_clusters=2)
    clusters = kmeanModel.fit_predict(X)
    distrotions.append(kmeanModel.inertia_)

    kmeanModel = KMeans(n_clusters=3)
    clusters = kmeanModel.fit_predict(X)
    distrotions.append(kmeanModel.inertia_)

    diff = np.abs(distrotions[len(distrotions) - 2] - distrotions[len(distrotions) - 1])
    diffs.append(diff)

    k = 4
    TH = 10

    while diffs[len(diffs) - 1] > TH:
    #while (k < 200) & (int(distrotions[len(distrotions) - 1]) > 100000):

        print(f'fitting for k = {k}')
        print(f'interia = {distrotions[len(distrotions) - 1]}')
        print(f'diffs = {diffs[len(diffs) - 1]}')
        print(f'TH = {distrotions[len(distrotions) - 2] - distrotions[len(distrotions) - 1]}')
        kmeanModel = KMeans(n_clusters=k)
        clusters = kmeanModel.fit_predict(X)
        distrotions.append(kmeanModel.inertia_)
        diff = np.abs(distrotions[len(distrotions) - 2] - distrotions[len(distrotions) - 1])
        diffs.append(diff)
        k+=1

    range_n_clusters = range(len(distrotions))
    plt.plot(range_n_clusters[len(range_n_clusters) - 10:], distrotions[len(range_n_clusters) - 10:], 'bx-')
    plt.xlabel('K')
    plt.yscale('linear')
    plt.ylabel('Distortion')
    plt.title("Elbow graph for Kmean")
    plt.savefig("Kmeans-Elbow.jpg")
    plt.show()

    plt.plot(diffs[len(diffs) - 10:], 'bx-')
    plt.xlabel('K')
    plt.yscale('linear')
    plt.ylabel('diffs')
    plt.title("Elbow graph for Kmean")
    plt.savefig("Kmeans-diffs.jpg")
    plt.show()

def yellowElbowK(data, model):
    visualizer = KElbowVisualizer(model, k =(2, 20))
    visualizer.fit(data)
    visualizer.show()

def clusterAnalysis(k,data,data2):
    kmeanModel = KMeans(n_clusters=k)
    clusters = kmeanModel.fit_predict(data)
    data2['cluster'] = clusters
    le = preprocessing.LabelEncoder()
    data2['EncodedSpecies'] = le.fit_transform(data2['Species'])

    for i in data2['cluster'].unique():

        d = data2.copy()
        d = d[d['cluster'] == i]
        sns.histplot(data=d, x="EncodedSpecies")
        plt.title(f'the number of species in cluster {i} is {d["EncodedSpecies"].unique().size} '
                  f'and the number of rows is {len(d)}')
        plt.show()

        x1 = d['A'].mean()
        x2 = d['C'].mean()
        x3 = d['T'].mean()
        x4 = d['G'].mean()

        nucleotide = [x1,x2,x3,x4]
        plt.bar(['A', 'C', 'T', 'G'], nucleotide)
        plt.title(f'single average nucleotide usage for cluster {i}')
        plt.show()

    print(f'The species in cluster 3 are {data2[data2["cluster"] == 3]["Species"].unique()}')
    print(f'The species in cluster 4 are {data2[data2["cluster"] == 4]["Species"].unique()}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_selection = "bicodon" #codon/ bicodon/ both
    mission = "clustering" #preprocess/ EDA / clustering

    if dataset_selection == 'codon':

        #Creating filltered & feature enriched csv
        if mission == 'preprocess':
            data = pd.read_table('o537-genbank_species.tsv')
            data = data.shift(periods = 1, axis = 1)
            data = FeatureEnrichment(data)
            data, data2 = Preprocess(data)
            data.to_csv("clustering_codon.csv")
            data2.to_csv("analysis_codon.csv")

        #Data EDA
        elif mission == 'EDA':
            data = pd.read_csv('clustering_codon.csv')
            Visualize(data)
            makeEntropyBins(data)
            makeCorrelationMatrix(data[['A','C','T','G']])
            makeCorrelationMatrix(data[[x[0]+x[1] for x in foo2('TGCA')]])
            makeCorrelationMatrix(data[[x[0]+x[1]+x[2] for x in foo3('TGCA')]])

        elif mission == 'clustering':
            data = pd.read_csv('clustering_codon.csv')
            data2 = pd.read_csv('analysis_codon.csv')
            Kmeans(data)
            #findOptimalK(data)
            clusterAnalysis(7,data,data2)

    elif dataset_selection == 'bicodon':

        if mission == 'preprocess':
            flag = 0
            data = pd.read_table('o537-genbank_Bicod.tsv', chunksize = 100000)
            for d in data:
                if flag == 0:
                    df = pd.DataFrame(columns = d.columns)
                    d = d[d['Organelle'] == 'genomic']
                    df = pd.concat([df,d])
                    flag = 1
                else:
                    d = d[d['Organelle'] == 'genomic']
                    df = pd.concat([df,d])
            data = pd.read_csv('bicodon.csv')
            data = FeatureEnrichmentBi(data)
            data = Preprocess(data)
            data.to_csv("clustering_bicodon.csv")
            #data2.to_csv("analysis_bicodon.csv")

        elif mission == 'EDA':
            #add code for EDA
            pass

        elif mission == 'clustering':
            data = pd.read_csv('clustering_bicodon.csv')
            yellowElbowK(data, KMeans())






