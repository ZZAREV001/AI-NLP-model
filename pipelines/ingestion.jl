module Ingestion

function load_data(file_path)
    # Load data from the file
    data = readlines(file_path)
    return data
end

end # module
