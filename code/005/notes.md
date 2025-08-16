RD
# issues:
Our framework's use of dist_2 (which flips sign) and dist_netw2 (squared term) is trying to approximate the separate slopes, but it's not equivalent to true LLR. This approximation gets worse with fewer observations because there's less data to "smooth out" the misspecification. -> that's why i just saved A_2 and B_2 results.