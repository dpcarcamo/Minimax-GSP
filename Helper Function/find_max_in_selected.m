function [maxValue, rowIndex, colIndex, truerowIndex, truecolIndex] = find_max_in_selected(matrix, selectedRows, selectedCols)
    % Extract the submatrix based on the specified rows and columns
    submatrix = matrix(selectedRows, selectedCols);
    
    % Find the maximum value and its indices in the submatrix
    [maxValue, linearIndex] = max(submatrix(:));
    [rowIndex, colIndex] = ind2sub(size(submatrix), linearIndex);
    
    truerowIndex = rowIndex;
    truecolIndex = colIndex;

    % Map the indices to the original matrix
    rowIndex = selectedRows(rowIndex);
    colIndex = selectedCols(colIndex);
end

