# COP22_VALIDATION — real-data gate for embedder choice

Slice: first 6,000 lines of `data/cop22.json`; originals clustered; min_topic_size=30, nr_topics=null, umap 15/10. Languages: en 900, fr 230, es 70, eu 21, ru 18, und 17.

## all-MiniLM-L6-v2

**2 topics**, 29 outliers.

| topic | n | lang mix | label | top c-TF-IDF terms |
|---|---|---|---|---|
| 0 | 947 | en:861, eu:21, und:17 | COP22 climate negotiations | climatechange, rt, p2, cop22, climate, change, tcot, 2a |
| 1 | 330 | fr:220, es:64, en:29 | Négociations climatiques | la, en, les, le, est, des, du, pour |

## paraphrase-multilingual-MiniLM-L12-v2

**9 topics**, 292 outliers.

| topic | n | lang mix | label | top c-TF-IDF terms |
|---|---|---|---|---|
| 0 | 373 | en:316, fr:27, es:12 | Climate science research | scientificpapers, climatechange, change, rt, climate, p2, 2a, tcot |
| 1 | 158 | fr:69, en:53, es:15 | COP22 key takeaways | la, en, marrakech, takeaway, recap, glimpses, messages, les |
| 2 | 83 | en:80, de:2, fr:1 | Trump administration climate policy | trump, trumps, epa, uncccf, obama, team, donald, p2 |
| 3 | 83 | en:70, fr:6, es:5 | Renewable energy in India | energy, renewables, solar, electricity, wind, india, power, renewable |
| 4 | 72 | fr:40, en:19, es:11 | Libros y artículos climáticos | googlealerts, web, libros, novelas, artculos, latest, thanks, leer |
| 5 | 68 | en:38, fr:24, es:3 | Jeunesse africaine à la COP22 | morocco, africa, maroc, marrakech, marrakesh, du, jeunes, des |
| 6 | 65 | en:64, fr:1 | Arctic ice melt | arctic, glacier, ice, north, pole, melting, temperatures, warm |
| 7 | 59 | en:32, eu:21, fr:4 | COP23 climate news updates | cop23cambioclimticoclimatechangechangementclimatique, va, news, ambiente, 12, stories, featuring, nov |
| 8 | 53 | en:32, fr:15, es:6 | U.S. and Paris Agreement | paris, agreement, laroport, internationale, amricaine, ouest, lactu, dix |

## Qwen/Qwen3-Embedding-0.6B

**14 topics**, 405 outliers.

| topic | n | lang mix | label | top c-TF-IDF terms |
|---|---|---|---|---|
| 0 | 110 | fr:68, en:20, qme:13 | Climate campaign updates | googlealerts, au, et, les, skypapers, la, lovemyclimate, tb |
| 1 | 104 | en:81, fr:12, es:6 | India’s renewable energy transition | energy, renewables, india, cars, pollution, solar, wind, half |
| 2 | 89 | en:39, fr:25, es:12 | COP22 Marrakech participation | marrakech, morocco, marrakesh, maroc, la, participation, uncop22, en |
| 3 | 86 | en:63, fr:12, es:6 | Climate news updates | latest, thanks, daily, journal, diario, actu, prtp, presse |
| 4 | 79 | en:71, es:3, tl:1 | Climate impacts on agriculture | scientificpapers, unfccc, water, featuring, farmers, diversity, nile, growing |
| 5 | 71 | en:69, de:2 | Trump and climate science | trump, trumps, epa, uncccf, team, donald, scientist, science |
| 6 | 59 | en:55, fr:3, ar:1 | Australian oil politics | oil, auspol, ambiente, nation, acting, act, qldpol, wapol |
| 7 | 56 | en:55, fr:1 | Wildlife and fisheries extinction | wildlife, extinction, fisheries, crocodile, driving, river, pictures, birds |
| 8 | 51 | en:39, fr:11, ru:1 | Brexit and environmental policy | 2016, stories, cette, ccs, wasnt, environmental, think, brexit |
| 9 | 51 | en:50, fr:1 | Arctic ice melt | arctic, glacier, ice, north, snow, pole, temperatures, glaciers |
| 10 | 47 | eu:21, en:18, fr:4 | Civil society at COP23 | cop23cambioclimticoclimatechangechangementclimatique, va, news, civsocprize, groups, eesc, enjoy, issue |
| 11 | 36 | en:29, fr:6, sv:1 | Démarche RSE et développement durable | highlights, conference, look, propose, dmarche, rse, tapes, developpementdurable |
| 12 | 32 | en:15, fr:10, und:3 | Forêts et biodiversité | forests, environnement, ni, klimawandel, biodiversity, thinklandscape, cop21, klimabewegung |
| 13 | 30 | fr:15, en:9, es:6 | Private climate investment | mouline, sad, dollars, smouline, del, private, startups, polo |
