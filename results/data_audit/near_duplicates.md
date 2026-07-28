# Near-duplicate audit — `data/images/`

Pool: **1578 images**; frozen gold subset: **500**.
Two 64-bit perceptual hashes (DCT pHash, gradient dHash) computed in numpy; a pair counts
as a near duplicate only when **both** are within Hamming 10/64.
Clusters are transitive closures (union-find) over those pairs.

## Byte-identical files (SHA-256)

- exact-duplicate clusters: **69**, images involved: **142**

## Threshold sweep (pool-wide)

| Hamming ≤ | pairs | clusters | images in clusters | pool near-dup rate | gold near-dup rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 92 | 78 | 163 | 0.1033 | 0.11 |
| 2 | 98 | 82 | 172 | 0.109 | 0.112 |
| 4 | 100 | 84 | 176 | 0.1115 | 0.114 |
| 6 | 100 | 84 | 176 | 0.1115 | 0.114 |
| 8 | 100 | 84 | 176 | 0.1115 | 0.114 |
| 10 | 100 | 84 | 176 | 0.1115 | 0.114 |
| 12 | 100 | 84 | 176 | 0.1115 | 0.114 |

## Headline (Hamming ≤ 10 on both hashes)

- pool near-duplicate rate: **0.1115** (176/1578 images in 84 clusters; largest cluster 3)
- frozen-gold near-duplicate rate: **0.114** (57/500 images)
- gold-internal pairs (both images in the gold 500): **11**
- **gold image near-duplicating a tuning-subset image**: **19** pairs, **24** distinct gold images

The last line is the leakage-relevant quantity: the frozen gold straddles `data/splits/`,
so a gold image that near-duplicates a prompt-tuning image is exposure that the
tuned-vs-held-out re-score (`split_leakage_summary.md`) would not otherwise catch.

## Clusters (Hamming ≤ 10), gold members marked `*`

1. (3) img_0150* img_0426 img_0640
2. (3) img_0163* img_1300 img_1301*
3. (3) img_0548 img_0745 img_1026*
4. (3) img_0574 img_0767 img_1121
5. (3) img_0793 img_1220 img_1522
6. (3) img_1127 img_1524* img_1525
7. (3) img_1178 img_1532 img_1536
8. (3) img_1527 img_1534 img_1541*
9. (2) img_0125* img_0400
10. (2) img_0176* img_0177*
11. (2) img_0182 img_0976*
12. (2) img_0184 img_0839
13. (2) img_0189* img_0661
14. (2) img_0204 img_0458*
15. (2) img_0205 img_0463
16. (2) img_0245 img_0489
17. (2) img_0248 img_0493
18. (2) img_0257 img_0874*
19. (2) img_0290 img_0527
20. (2) img_0295 img_0886*
21. (2) img_0328* img_0564*
22. (2) img_0332* img_0333
23. (2) img_0360 img_1533*
24. (2) img_0363* img_1537*
25. (2) img_0382 img_1314
26. (2) img_0396 img_1520
27. (2) img_0403 img_1289*
28. (2) img_0448* img_0838
29. (2) img_0497 img_0706
30. (2) img_0498* img_1005
31. (2) img_0545 img_0734*
32. (2) img_0556 img_0750
33. (2) img_0563* img_0761*
34. (2) img_0576 img_0577
35. (2) img_0585* img_0648*
36. (2) img_0612* img_0800
37. (2) img_0619 img_0804
38. (2) img_0653 img_1455
39. (2) img_0666 img_1331*
40. (2) img_0714 img_1202*
41. (2) img_0744* img_1472*
42. (2) img_0754 img_0909
43. (2) img_0776* img_0831*
44. (2) img_0792* img_0939
45. (2) img_0805 img_1287*
46. (2) img_0836 img_1327
47. (2) img_0852 img_0987
48. (2) img_0871 img_1004
49. (2) img_0900 img_1235*
50. (2) img_0971 img_1456
51. (2) img_1029 img_1478
52. (2) img_1070 img_1140
53. (2) img_1168 img_1470
54. (2) img_1234 img_1498
55. (2) img_1255* img_1256
56. (2) img_1261* img_1262
57. (2) img_1263 img_1264
58. (2) img_1273 img_1274
59. (2) img_1279* img_1280
60. (2) img_1296 img_1297
61. (2) img_1298 img_1299
62. (2) img_1304 img_1305
63. (2) img_1316 img_1317*
64. (2) img_1325 img_1326
65. (2) img_1422 img_1423
66. (2) img_1424 img_1425
67. (2) img_1426 img_1427
68. (2) img_1452 img_1453
69. (2) img_1475 img_1476
70. (2) img_1479* img_1480
71. (2) img_1481 img_1482
72. (2) img_1483* img_1484*
73. (2) img_1485 img_1486
74. (2) img_1490 img_1491*
75. (2) img_1514* img_1515
76. (2) img_1516 img_1517*
77. (2) img_1518* img_1519*
78. (2) img_1530* img_1531
79. (2) img_1552 img_1554
80. (2) img_1560 img_1569*
81. (2) img_1561* img_1562*
82. (2) img_1563 img_1570*
83. (2) img_1566 img_1575*
84. (2) img_1573 img_1574*
