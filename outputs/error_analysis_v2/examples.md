# Error analysis examples (Mistral v2 predictions, fine-grained buckets)

## Dataset: squad_v2  Chunker: recursive_256

### Retrieval Failure (3 examples shown)

- **Q:** What entity owns V/Line?
  **Gold:** Victorian Government | the Victorian Government | Victorian Government
  **Pred:** unanswerable
  **Retrieved titles:** European_Union_law, European_Union_law, Packet_switching, Packet_switching
  **F1:** 0.000  **Recall@4:** 0.000

- **Q:** Who was Ralph in charge of being at war with?
  **Gold:** the Welsh | the Welsh | the Welsh
  **Pred:** unanswerable
  **Retrieved titles:** French_and_Indian_War, French_and_Indian_War, French_and_Indian_War, French_and_Indian_War
  **F1:** 0.000  **Recall@4:** 0.000

- **Q:** What was the source of the mistake?
  **Gold:** the WWF report | the IPCC from the WWF report | the WWF report
  **Pred:** unanswerable
  **Retrieved titles:** 1973_oil_crisis, Packet_switching, French_and_Indian_War, French_and_Indian_War
  **F1:** 0.000  **Recall@4:** 0.000


### False Refusal (3 examples shown)

- **Q:** Was the Use of the DATANET 1 name correct 
  **Gold:** use of the name was incorrect all these services were managed by the same people within one department of KPN contributed to the confusion | the name was incorrect | Dutch PTT Telecom
  **Pred:** unanswerable
  **Retrieved titles:** Packet_switching, Packet_switching, Packet_switching, Packet_switching
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** What does critically tapered mean? 
  **Gold:** all angles remain the same | all angles remain the same | all angles remain the same
  **Pred:** unanswerable
  **Retrieved titles:** Geology, Civil_disobedience, Intergovernmental_Panel_on_Climate_Change, Civil_disobedience
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** Where did the Grand Canal start?
  **Gold:** southern China | southern China | southern China
  **Pred:** unanswerable
  **Retrieved titles:** Rhine, Yuan_dynasty, French_and_Indian_War, Rhine
  **F1:** 0.000  **Recall@4:** 1.000


### Format Verbose (3 examples shown)

- **Q:** What is thought to have happened to the y. pestis that caused the black death?
  **Gold:** may no longer exist | may no longer exist | may no longer exist
  **Pred:** Y. pestis may no longer exist
  **Retrieved titles:** Black_Death, Black_Death, Black_Death, Black_Death
  **F1:** 0.800  **Recall@4:** 1.000

- **Q:** What is the nature of the relationship between T-cells and vitamin D?
  **Gold:** symbiotic relationship | symbiotic | symbiotic relationship
  **Pred:** symbiotic relationship with vitamin D
  **Retrieved titles:** Immune_system, Immune_system, Immune_system, Immune_system
  **F1:** 0.571  **Recall@4:** 1.000

- **Q:** What nuclear forces only act at short distances?
  **Gold:** strong and weak | strong and weak forces | strong and weak | strong and weak forces
  **Pred:** Strong and weak forces act only at very short distances
  **Retrieved titles:** Force, Force, Force, Force
  **F1:** 0.571  **Recall@4:** 1.000


### Format Terse (2 examples shown)

- **Q:** What supplanted Frame Relay and X.25 
  **Gold:** supplanted by the Internet Protocol (IP) at the network layer, and the Asynchronous Transfer Mode (ATM) and or versions of Multi-Protocol Label Switching | Internet Protocol (IP) | Internet Protocol
  **Pred:** Internet Protocol (IP), Asynchronous Transfer Mode (ATM), and versions of Multi-Protocol Label Switching (
  **Retrieved titles:** Packet_switching, Packet_switching, Packet_switching, Packet_switching
  **F1:** 0.788  **Recall@4:** 1.000

- **Q:** How many refugees emigrated to the Dutch Republic?
  **Gold:** an estimated total of 75,000 to 100,000 people | 75,000 to 100,000 | 75,000 to 100,000
  **Pred:** 75,000
  **Retrieved titles:** Huguenot, Huguenot, Huguenot, Huguenot
  **F1:** 0.500  **Recall@4:** 1.000


### Paraphrase (3 examples shown)

- **Q:** Contracts must be designed to ensure what?
  **Gold:** that each side is capable of performing the obligations set out | that each side is capable of performing the obligations set out | each side is capable of performing the obligations set out
  **Pred:** Contracts must ensure each side is capable of performing obligations
  **Retrieved titles:** Construction, Construction, Construction, Construction
  **F1:** 0.737  **Recall@4:** 1.000

- **Q:** How much Saharan dust remains in the air over the Amazon each year?
  **Gold:** 132 million tons | 132 million tons | 132 million tons
  **Pred:** 27.7 million tons
  **Retrieved titles:** Amazon_rainforest, Amazon_rainforest, Amazon_rainforest, Amazon_rainforest
  **F1:** 0.667  **Recall@4:** 1.000

- **Q:** What are the three sources of European Union law?
  **Gold:** primary law, secondary law and supplementary law | primary law, secondary law and supplementary law | primary law, secondary law and supplementary law | primary law, secondary law and supplementary law
  **Pred:** Three sources of European Union law: primary law, secondary law, supplementary law
  **Retrieved titles:** European_Union_law, European_Union_law, European_Union_law, European_Union_law
  **F1:** 0.632  **Recall@4:** 1.000


### Partial Answer (3 examples shown)

- **Q:** Why does a lower level of economic growth occur due to high-end consumption?
  **Gold:** human capital is neglected | a lower level of economic utility in society | human capital is neglected
  **Pred:** Lower level of economic growth due to high-end consumption
  **Retrieved titles:** Economic_inequality, Economic_inequality, Economic_inequality, Economic_inequality
  **F1:** 0.500  **Recall@4:** 1.000

- **Q:** According to Wilson's theorem, what factorial must be divisible by p if some integer p > 1 is to be considered prime?
  **Gold:** (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1
  **Pred:** (p - 1)! + 1 is divisible by p
  **Retrieved titles:** Prime_number, Prime_number, Prime_number, Prime_number
  **F1:** 0.545  **Recall@4:** 1.000

- **Q:** What aspect of Western medicine did the Chinese dislike?
  **Gold:** its humoral system | its humoral system | humoral system
  **Pred:** Chinese physicians opposed Western medicine due to its contradiction with yin-yang and wuxing philosophy
  **Retrieved titles:** Yuan_dynasty, Yuan_dynasty, Yuan_dynasty, Yuan_dynasty
  **F1:** 0.118  **Recall@4:** 1.000


### Wrong Answer (3 examples shown)

- **Q:** How many counties initially made up the definition of southern California?
  **Gold:** seven | seven | seven
  **Pred:** The context does not provide a specific number of counties initially making up southern California according to the historical attempts for separate statehood or territorial status
  **Retrieved titles:** Southern_California, Southern_California, Southern_California, Southern_California
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** Where can Aeolian sand with a number of dunes be found?
  **Gold:** plain Vistula terraces | highest terrace | highest terrace
  **Pred:** Warsaw
  **Retrieved titles:** Warsaw, Amazon_rainforest, Geology, Geology
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** Which timeline is further expanded in the fourth scale?
  **Gold:** Quaternary | Quaternary | Quaternary
  **Pred:** further expanded in the fourth scale
  **Retrieved titles:** Geology, Imperialism, Intergovernmental_Panel_on_Climate_Change, Geology
  **F1:** 0.000  **Recall@4:** 1.000


## Dataset: hotpot_qa  Chunker: recursive_256

### Retrieval Failure (0 examples shown)

_No examples in this bucket._

### False Refusal (3 examples shown)

- **Q:** Robert Earl Holding owned an oil company that was originally founded by who?
  **Gold:** Harry F. Sinclair
  **Pred:** unanswerable
  **Retrieved titles:** Robert Holding, 101 Ranch Oil Company, Skelly Oil, Sinclair Oil Corporation
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** Were the films Crazy Love and Welcome to Nollywood both released in 2008?
  **Gold:** no
  **Pred:** unanswerable
  **Retrieved titles:** Welcome to Nollywood, Crazy Love (2007 film), Crazy Love (Michael Bublé album), This Crazy Love
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** When did the car depicted on the cover of Pentastar: In the Style of Demons cease production?
  **Gold:** 1974
  **Pred:** unanswerable
  **Retrieved titles:** Pentastar: In the Style of Demons, Zeitoper, Plymouth Barracuda, Maiwand Lion
  **F1:** 0.000  **Recall@4:** 1.000


### Format Verbose (3 examples shown)

- **Q:** What Kentucky county has a population of 60,316 and features the Lake Louisvilla neighborhood?
  **Gold:** Oldham County
  **Pred:** Oldham County, Kentucky
  **Retrieved titles:** Lake Louisvilla, Louisville, Kentucky County, Virginia, Oldham County, Kentucky, Casey County, Kentucky
  **F1:** 0.800  **Recall@4:** 1.000

- **Q:** After his curacy at the village that is a suburb of Scunthorpe, who was Industrial Chaplain to the Bishop of Lincoln?
  **Gold:** Dudman
  **Pred:** Bill Dudman
  **Retrieved titles:** John Gathercole, Bill Dudman, Randolph Wise, William Everingham
  **F1:** 0.667  **Recall@4:** 0.500

- **Q:** Politician Lyman Sherwood was born in what New York county?
  **Gold:** Rensselaer County
  **Pred:** Rensselaer County, New York
  **Retrieved titles:** Lyman Sherwood, Sherwood Equal Rights Historic District, Charles D. Sherwood, Slocum and Hannah Howland House
  **F1:** 0.667  **Recall@4:** 0.500


### Format Terse (2 examples shown)

- **Q:** Pacific Mozart Ensemble performed which German composer's Der Lindberghflug in 2002?
  **Gold:** Kurt Julian Weill
  **Pred:** Kurt Weill
  **Retrieved titles:** The Flight Across the Ocean, Pacific Mozart Ensemble, Pacific Mozart Ensemble, Martin Boykan
  **F1:** 0.800  **Recall@4:** 0.500

- **Q:** What commentator and author questioned the validity of CORE's non-profit status?
  **Gold:** Rachel Anne Maddow
  **Pred:** Rachel Maddow
  **Retrieved titles:** Center for Organizational Research and Education, Brown Association for Cooperative Housing, American Bladesmith Society, York City War Memorial
  **F1:** 0.800  **Recall@4:** 0.500


### Paraphrase (0 examples shown)

_No examples in this bucket._

### Partial Answer (1 examples shown)

- **Q:** If the Charhki  Dadri crash was less dangerous than the Tenerife airport disaster, which occured firat? 
  **Gold:** On March 27, 1977, two Boeing 747 passenger jets, KLM Flight 4805 and Pan Am Flight 1736, collided on the runway at Los Rodeos Airport (now Tenerife North Airport)
  **Pred:** The Tenerife airport disaster occurred first
  **Retrieved titles:** Charkhi Dadri mid-air collision, 1977 in aviation, Tenerife airport disaster, Jacob Veldhuyzen van Zanten
  **F1:** 0.121  **Recall@4:** 1.000


### Wrong Answer (3 examples shown)

- **Q:** Which mountain is taller, Gasherbrum II or Langtang Ri?
  **Gold:** Gasherbrum II
  **Pred:** taller than Langtang Ri (72
  **Retrieved titles:** Gasherbrum II, Langtang Ri, Gasherbrum I, Gasherbrum III
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** What years did Jose Gonzalo Rodriguez Gacha and other leaders of fthe Medallin Cartel operate in Boliva, Colombia, Central America, Peru, the United States, Canada, and Europe?
  **Gold:** 1970s and 1980s
  **Pred:** 1970s-1980s (in various countries)
  **Retrieved titles:** José Gonzalo Rodríguez Gacha, Juan Carlos Ramírez Abadía, Medellín Cartel, Wilber Varela
  **F1:** 0.000  **Recall@4:** 1.000

- **Q:** What nationality was Oliver Reed's character in the film Royal Flash?
  **Gold:** Prussian
  **Pred:** German
  **Retrieved titles:** Royal Flash (film), Royal Flash, Oliver Reed, Harry Flashman
  **F1:** 0.000  **Recall@4:** 0.500

