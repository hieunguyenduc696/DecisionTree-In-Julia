using Pkg
Pkg.add("StatsBase") # generate unique value using sample method
Pkg.add("CSV")
Pkg.add("DataFrames")
# import statement
using CSV
using DataFrames
using StatsBase

#=
    splitting train & test data with the ratio 2/3 & 1/3
    accept two params: 
        - DataFrame data
        - test_size: float (ex: 0.3333) or fixed value: 50
=#
function train_test_split(data, test_size)
    if typeof(test_size) == Float64
        test_size = Int64(round(test_size * nrow(data)))
    end
    test_indicies = sample(1:nrow(data), test_size, replace = false)
    train_df = filter(row -> !issubset(row.Id, test_indicies), data)
    test_df = filter(row -> issubset(row.Id, test_indicies), data)
    train_df = select!(train_df, Not(:Id))
    test_df = select!(test_df, Not(:Id))
    return train_df, test_df
end

#=
    check whether data is pure or not
    all rows have the same type of flower => true
=#
function check_purity(data)
    label_column = data[1:end, end]
    unique_classes = unique!(label_column)
    return length(unique_classes) == 1
end

#= 
    classify data (setosa, virginica, versicolor) base on the value of appearance in data
    Ex: [setosa: 10, virginica: 2, versicolor: 3] => return setosa
=#
function classify_data(data)
    label_column = data[1:end, end]
    unique_classes = unique!(label_column)
    count_array = [count(==(x), data) for x in label_column]
    index = findall(isequal(maximum(count_array)), count_array)
    classification = unique_classes[index]
    return classification
end

# get potential splits from data
function get_potential_splits(data)
    potential_splits = Dict()
    n_columns = size(data)[2]
    for column_index = 1:n_columns-1
        potential_splits[column_index] = []
        values = data[:,column_index]
        unique_values = unique!(values)
        for index = 1:length(unique_values)
            if index != 1
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                push!(potential_splits[column_index], potential_split)
            end
        end
        potential_splits[column_index] = sort(potential_splits[column_index])
    end
    return potential_splits
end

# divide data into two sub tree
function split_data(data, split_column, split_value)
    split_column_values = data[:, split_column]
    data_below = filter(row->(row[split_column] < split_value), data)
    data_above = filter(row->(row[split_column] >= split_value), data)
    return Array(data_below), Array(data_above)
end

# calculate entropy of data
function calculate_entropy(data)
    label_column = data[:, end]
    unique_classes = unique!(label_column)
    count_array = [count(==(x), data) for x in unique_classes]
    probabilities = count_array / sum(count_array)
    entropy = 0
    for x in probabilities
        entropy += x * -log2(x)
    end
    return entropy
end

# calculate the overall entropy
function calculate_overall_entropy(data_below, data_above)
    n_data_points = length(data_below) + length(data_above)
    p_data_below = length(data_below) / n_data_points
    p_data_above = length(data_above) / n_data_points
    overall_entropy = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))
    return overall_entropy
end

# choose the best split from potential splits
function determine_best_split(data, potential_splits)
    overall_entropy = 999
    best_split_column = 0
    best_split_value = 0
    for key in keys(potential_splits)
        for val in potential_splits[key]
            data_below, data_above = split_data(data, key, val)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy
                overall_entropy = current_overall_entropy
                best_split_column = key
                best_split_value = val
            end
        end
    end
    return best_split_column, best_split_value 
end

function decision_tree_algorithm(df, counter = 0, min_samples = 2, max_depth = 5)
    # data preparation
    if counter == 0
        data = Array(df)
        global COLUMN_HEADERS = names(df)[1:end]
    else data = df
    end
    
    # base cases
    if ((typeof(data) != DataFrame) && (length(data) < min_samples)) || (check_purity(data)) || 
        ((typeof(data) == DataFrame) && (size(df)[1] * size(df)[2] < min_samples)) || (counter == max_depth)
        classification = classify_data(data)
        return classification
    # recursive part
    else
        counter += 1
        # helper function
        potential_splits = get_potential_splits(data)
        if typeof(data) != DataFrame
            split_column, split_value = determine_best_split(DataFrame(data,:auto), potential_splits)
            data_below, data_above = split_data(DataFrame(data,:auto), split_column, split_value)
            data = Array(data)
        else 
            split_column, split_value = determine_best_split(data, potential_splits)
            data_below, data_above = split_data(data, split_column, split_value)
        end
        # instantiate sub tree
        feature_name = COLUMN_HEADERS[split_column]
        question = string(feature_name, " <= ", split_value)
        sub_tree = Dict(question => [])
        
        # find answer (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        if(yes_answer == no_answer)
            sub_tree = yes_answer
        else
            push!(sub_tree[question], yes_answer)
            push!(sub_tree[question], no_answer)
        end
        return sub_tree
    end
end

# classify the tree after calling main algorithm
function classify_example(example, tree)
    question = collect(keys(tree))[1]
    feature_name, comparison_operator, value = split(question, " ")
    if example[feature_name] <= parse(Float64,string(value))
        answer = tree[question][1]
    else
        answer = tree[question][2]
    end
    # base case
    if typeof(answer) != Dict{String, Vector{Any}}
        return answer
    else
        # recursive part
        residual_tree = answer
        return classify_example(example, residual_tree)
    end
end

# classify along with calculating the accuracy of main algorithm
function calculate_accuracy(df, tree)
    df.classification = [classify_example(example, tree)[1] for example in eachrow(df)]
    df.correct_classification = 
    [df.classification[iter] == df.species[iter] for iter in 1:length(df.classification)]
    accuracy = mean(df.correct_classification)
    return accuracy
end

function readFile()
    csv_file = "Iris.csv" # input filename
    df = CSV.read(csv_file, DataFrame)
    return df
end

function main() 
    # input
    df = readFile()

    # processing
    test_proportion = 0.3333 # define the size of test set 
    train_df, test_df = train_test_split(df, test_proportion)

    # helper varibales
    counter = 0 # to know whether in the first loop or not
    min_samples = 60 # limit the size of sample, add one more condition in base case in main algorithm (Decision Tree)
    max_depth = 3 #= limit num of layers of the output tree in main algorithm (Decision Tree), the higher this value is
    the more accuration the result gets =#
    tree = decision_tree_algorithm(train_df, counter, min_samples, max_depth)

    # output
    print(calculate_accuracy(test_df, tree))
end

main()