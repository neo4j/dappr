from cProfile import label
from collections import defaultdict

from matplotlib.figure import Figure
from candidate_selection.benchmarking.benchmark_reader import BenchmarkReader, BenchmarkResult, Parameters, Statistics
import matplotlib.pylab as plt
from matplotlib import rcParams
from dataset.dataset_reader import Datasets
import numpy as np
from statistics import mean
from itertools import groupby


class BenchmarkVisualizer:

    def __init__(self, file_name="benchmarks.txt") -> None:
        self.__benchmarks: list[BenchmarkResult] = BenchmarkReader(file_name).read()
        self.all_algorithm_names = list(set(map(lambda x: x["method"], self.__benchmarks)))
        self.all_dataset_names = list(set(map(lambda x: x["dataset"], self.__benchmarks)))

    def print_time_table(self,
        algorithm_names: list[str] | None,
    ):
        data = self.__benchmarks

        if not algorithm_names:
            algorithm_names = self.all_algorithm_names
        
        datasets = [
            Datasets.US_AIR,
            Datasets.YEAST,
            Datasets.MOVIE_LENS,
            Datasets.FACEBOOK1,
            Datasets.OGB_DDI,
            Datasets.POWER,
            Datasets.ROUTER,
            Datasets.HS_PROTEIN,
            Datasets.DBLP,
            Datasets.ARXIV,
            Datasets.MATH_OVERFLOW,
            Datasets.FACEBOOK2,
            Datasets.REDDIT,
            Datasets.EPINIONS,
            Datasets.ENRON,
            Datasets.OGB_COLLAB,
            Datasets.DIGG,
            Datasets.AMAZON,
            Datasets.ROADNET_PA,
        ]

        best_recall_on_datasets_counts = defaultdict(lambda: 0)
        best_tplp_on_datasets_counts = defaultdict(lambda: 0)

        for dataset in datasets:
            dataset = dataset.name
            outstr = str(dataset).replace("_", "\_") + " & "

            for algo in algorithm_names:
                try:
                    h_1 = list(filter(lambda x: x["dataset"] == dataset and x["method"] == algo and (x["parameters"]["h"] == 1), data))[0]
                    outstr += str(round(h_1["statistics"]["time"], 1)) + " & "
                except Exception as e:
                    outstr += "XX & "
                try:
                    h_10 = list(filter(lambda x: x["dataset"] == dataset and x["method"] == algo and (x["parameters"]["h"] == 10), data))[0]
                    outstr += str(round(h_10["statistics"]["time"], 1)) + " & "
                except Exception as e:
                    outstr += "XX & "
                

            outstr = outstr[:-2] # Remove last &
            print(outstr + " \\\\\n\hline")
    

    def print_best_results(self,
        algorithm_names: list[str] | None,
    ):
        data = self.__benchmarks

        if not algorithm_names:
            algorithm_names = self.all_algorithm_names

        datasets = [
            Datasets.US_AIR,
            Datasets.YEAST,
            Datasets.MOVIE_LENS,
            Datasets.FACEBOOK1,
            Datasets.OGB_DDI,
            Datasets.POWER,
            Datasets.ROUTER,
            Datasets.HS_PROTEIN,
            Datasets.DBLP,
            Datasets.ARXIV,
            Datasets.MATH_OVERFLOW,
            Datasets.FACEBOOK2,
            Datasets.REDDIT,
            Datasets.EPINIONS,
            Datasets.ENRON,
            Datasets.OGB_COLLAB,
            Datasets.DIGG,
            Datasets.AMAZON,
            Datasets.ROADNET_PA,
        ]

        best_avg_recall_on_datasets_counts = defaultdict(lambda: 0)
        best_avg_tplp_on_datasets_counts = defaultdict(lambda: 0)

        for dataset in datasets:
            dataset = dataset.name
            # print("Dataset:", dataset)

            avg_recall = defaultdict(lambda: 0)
            avg_tplp = defaultdict(lambda: 0)

            for h in range(1,11):
                best_recall = 0.0
                best_tplp = 0.0

                for algo in algorithm_names:
                    # Add top performers
                    try:
                        result = list(filter(lambda x: x["dataset"] == dataset and x["method"] == algo and x["parameters"]["h"] == h, data))
                        if len(result) == 0:
                            continue
                        result = result[0]
                        # for result in filter(lambda x: x["dataset"] == dataset and x["method"] == algo and x["parameters"]["h"] == h, data):
                        avg_recall[algo] += result["statistics"]["recall"]
                        # avg_tplp[algo] += result["statistics"]["tplp"]

                        if best_recall < result["statistics"]["recall"]:
                            best_recall = result["statistics"]["recall"]

                        # # if best_tplp < result["statistics"]["tplp"]:
                            # # best_tplp = result["statistics"]["tplp"]
                    except Exception as e:
                        print("fam")
                        raise e

            # print("avg_recall", avg_recall)
            print("avg_tplp", avg_tplp)
            best_avg_recall = sorted(avg_recall.items(), key=lambda x: x[1], reverse=True)[0]
            # # best_avg_tplp = sorted(avg_tplp.items(), key=lambda x: x[1], reverse=True)[0]
            print('All avgs', dataset, [(algo, round(avg_recall[algo]/10, 2)) for algo in algorithm_names])
            
            avg_recall = list(sorted(map(lambda x: (x[0], x[1]/10), avg_recall.items()), key=lambda x: x[1], reverse=True))
            print(avg_recall)

            if avg_recall[0][0] == "Dappr":
                print("Dappr compared to second best:", round((avg_recall[0][1] - avg_recall[1][1])/avg_recall[1][1], 2))
            else:
                # Find dappr
                dappr = list(filter(lambda x: x[0] == "Dappr", avg_recall))[0]
                print("Dappr compared to best:", round((dappr[1] - avg_recall[0][1])/avg_recall[1][1], 2))


            print('Best avg: ',avg_recall[0])
            best_avg_recall_on_datasets_counts[best_avg_recall[0]] += 1
            # # best_avg_tplp_on_datasets_counts[best_avg_tplp[0]] += 1


        print("AVERGAE recall scores: ", list(best_avg_recall_on_datasets_counts.items()))
        # print("AVERGAE tplp scores: ", list(best_avg_tplp_on_datasets_counts.items()))
            


    def plot(self,
        x_name: str,
        y_name: str,
        dataset: Datasets,
        algorithm_names: list[str] | None,
        show_grid = True,
        show_title = True,
        log_scale = False,
        y_from_zero=False,
        legend=True,
        font_size=None,
    ) -> Figure:
        if font_size:
            rcParams.update({'font.size': font_size})

        if not algorithm_names:
            algorithm_names = self.all_algorithm_names
    
        xss, yss, labels = self.get_xy(x_name, y_name, dataset, algorithm_names)
        figure, axes = plt.subplots()

        self.configure_axes(x_name, y_name, axes, xss, yss, y_from_zero=y_from_zero)
        if show_grid: 
            self.show_grid(axes)

        if log_scale:
            axes.set_yscale("log")
            
        for xs, ys, label in zip(xss, yss, labels):
            try:
                linestyle, marker, color = self.get_line_style(label)
                axes.plot(xs, ys, c=color, linestyle=linestyle, label=label, marker=marker)
            except:
                continue

        if show_title:
            self.set_title(dataset, algorithm_names, axes)
        if len(algorithm_names) > 1:
            # Order legend alphabetically by algorithm_name
            handles, labels = axes.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

            if legend:
                if len(algorithm_names) < 4:
                    axes.legend(handles, labels)

                else:
                    box = axes.get_position()
                    axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # Put a legend to the right of the current axis
                    axes.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5)) # , ncol=len(labels)) # Get horizontal labels

        return figure
            

    def configure_axes(self, x_name: str, y_name: str, axes, xss: list[list], yss: list[list], y_from_zero=False):
        all_xs = list()
        all_ys = list()
        for xs, ys in zip(xss, yss):
             all_xs.extend(xs)
             all_ys.extend(ys)

        axes.set_xlabel(x_name)
        axes.set_ylabel(y_name)
        axes.set_xticks(all_xs)
        if len(all_ys) < 6:
            axes.set_yticks(all_ys)

        if y_from_zero:
            axes.set_ylim(bottom=0)

    def show_grid(self, axes):
        plt.rc('grid', linestyle=':', linewidth=1)
        axes.grid(True)

    def set_title(self, dataset: Datasets, algorithm_names: list[str], axes):
        title = ""
        if algorithm_names and len(algorithm_names) == 1:
            title += algorithm_names[0]
        if dataset:
            if title != "":
                title += " on "

            title += dataset.name
        
        if title != "":
            axes.set_title(title)

    # Used for relabeling specific algos
    def get_label_name(self, algo_name: str, data: list):
        algo_data = list(filter(lambda x: x["method"] == algo_name, data))

        return algo_name

    def get_xy(self, x_name: str, y_name: str, dataset: Datasets, algorithm_names: list[str] | None = None):
        data = self.filter_data(dataset, algorithm_names)
        if len(data) == 0: 
            raise "No data"
            
        xss = list()
        yss = list()
        labels = list()
        labels = list()
        
        for algo in algorithm_names:
            xs = list()
            ys = list()
            label_name = self.get_label_name(algo, data)
            labels.append(label_name)
            for result in list(filter(lambda x: x["method"] == algo, data)):
                xs.append(result["parameters"][x_name])
                ys.append(result["statistics"][y_name])
            # xs, ys = zip(*sorted(zip(xs, ys), key=lambda t: t[0]))
            xss.append(xs)
            yss.append(ys)

        xss, yss = self.meanify(xss, yss)

        return xss, yss, labels

    def meanify(self, xss, yss):
        """
        In case of multiple occurences of an x, converts all corresponding ys to mean(ys).
        """
        new_xss = list()
        new_yss = list()
        for xs, ys in zip(xss, yss):
            grouper = groupby(list(zip(xs, ys)), key=lambda x: x[0])
            xs_ys = [[x, mean(yi[1] for yi in y)] for x,y in grouper]
            new_xs = [ x for x, y in xs_ys ]
            new_ys = [ y for x, y in xs_ys ]
            new_xss.append(new_xs)
            new_yss.append(new_ys)
        xss = new_xss
        yss = new_yss
        return xss, yss

    def filter_data(self, dataset, algorithm_names):
        data = self.__benchmarks
        if algorithm_names:
            data = list(filter(lambda x: x["method"] in algorithm_names, data))
        if dataset:
            data = list(filter(lambda x: x["dataset"] == dataset.name, data))
        data = self.__add_missing_data(data)
        data = list(sorted(data, key=lambda x: x["parameters"]["h"], reverse=True))
        return data

    def __add_missing_data(self, results):
        for result in results:
            if "statistics" in result and "scans_per_found_edge" in result["statistics"]:
                scans_per_found_edge = result["statistics"]["scans_per_found_edge"]
                result["statistics"]["TPLP"] = 1 / scans_per_found_edge if scans_per_found_edge > 0 else 0
        return results

    def get_line_style(self, label: str):
        if label == "Dappr":
            return "--", "v", "blue"
        elif label == "Bfs":
            return ":", "^", "red"
        elif label == "KnnSimple":
            return "-.", "v", "magenta"
        elif label.startswith("Knn"):
            return "-.", "v", "magenta"
        elif label == "LinkWaldo":
            return "-.", "o", "limegreen"
        

