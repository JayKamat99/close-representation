import numpy as np

def pos_in_circular_interval(number, interval):
    interval = [x % (2 * np.pi) for x in interval]
    if interval[0] > interval[1]:  # Interval crosses (2pi, 0)
        intv = [interval[0] - 2 * np.pi, interval[1]]
        if number > interval[0]:  
            num = number - 2 * np.pi  
        else:  
            num = number
    else:
        intv = interval
        num = number
    
    if num < intv[0] or num > intv[1]:
        return 0
    return (num - intv[0]) / (intv[1] - intv[0])

def check_percentage_folded_corners(indFC, vec, folds):
    if indFC == 0: #if it's the first corner
        ant = len(vec)-1
    else:
        ant = indFC - 1
    
    if indFC == len(vec)-1:
        post = 0
    else:
        post = indFC + 1
    
    pos1 = pos_in_circular_interval(folds[0], [vec[ant], vec[indFC]])
    pos2 = pos_in_circular_interval(folds[0], [vec[indFC], vec[post]])
    
    if pos2 == 0:
        pos2 = pos_in_circular_interval(folds[1], [vec[indFC], vec[post]])
    
    percent = 0
    sOut = ""
    if pos1 != 0:
        percent = 100 * (1 - pos1)
        sOut += f"\nSide {ant+1}{indFC+1} is {percent:.2f}% folded"
    
    if pos2 != 0:
        if percent == 0:
            percent = 100 * pos2
        else:
            percent = [percent, 100 * pos2]
        sOut += f"\nSide {indFC+1}{post+1} is {100 * pos2:.2f}% folded"
    
    return percent, sOut

def num_elem(corner, vec):
    return vec.index(corner)  +1 #To compensat that starts with 0

def construct_sentence(vec, folds):
    folds = [x % (2 * np.pi) for x in folds]
    vec = [x % (2 * np.pi) for x in vec]
    if folds[0] < folds[1]:
        folded_corners = [v for v in vec if folds[0] < v < folds[1]]
    else:
        folded_corners = list(set(vec) - {v for v in vec if folds[1] < v < folds[0]})
    
    inds_folded = [num_elem(c, vec) for c in folded_corners]
    
    if len(vec) > 20:
        s1 = "Cloth is circular"
        percents = 100. * len(folded_corners) / len(vec)
        s3 = f"\nCloth is {percents:.2f}% folded."
        s4 = "\nCloth is folded in half" if 40 < percents < 60 else ""
    else:
        s1 = f"Cloth has {len(vec)} corners."
        s2 = f"\nCorner {inds_folded[0]} is folded." if len(inds_folded) == 1 else f"\nCorners {inds_folded} are folded."
        
        percents = [check_percentage_folded_corners(c-1, vec, folds) for c in inds_folded]
        s3 = "".join(p[1] for p in percents)
        percents = [p[0] for p in percents]
        
        non_zero_percents = [p for p in np.ravel(percents) if p != 0]
        similarity_percent = np.mean(np.abs(np.diff(non_zero_percents))) if len(non_zero_percents) > 1 else 0
        
        s4 = ""
        if len(inds_folded) == len(vec) // 2:
            s4 = "\nHalf of the corners are folded."
            if similarity_percent < 10: #small similarity means percentages of folded sides are similar
                mean_percent = np.mean(non_zero_percents)
                if 40 < mean_percent < 60:
                    s4 = "\nCloth is folded in half"
                else:
                    s4 = "\nOne side folded"
            elif 60 < max(non_zero_percents) and 20 < min(non_zero_percents):
                s4 = "\nFolded in half-askew."
            else:
                s4 = "\nOne side askew folded."
        
        if len(inds_folded) == 1 and similarity_percent < 10:
            mean_percent = np.mean(non_zero_percents)
            if 90 < mean_percent < 100:
                s4 = "\nDiagonally folded in half"
                s3 = ""
            else:
                s3 += f"\nCorner is {mean_percent:.2f}% folded."
    
    return s1 + (s2 if 's2' in locals() else '') + s4 + s3

def get_semantic_label(featureVector):
    corners = featureVector[0].tolist()
    fold = featureVector[1].tolist()
    return (construct_sentence(corners, fold))