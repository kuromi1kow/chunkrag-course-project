# Error analysis examples (v2 predictions)

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


### Formatting Mismatch (3 examples shown)

- **Q:** What is thought to have happened to the y. pestis that caused the black death?
  **Gold:** may no longer exist | may no longer exist | may no longer exist
  **Pred:** Y. pestis may no longer exist
  **Retrieved titles:** Black_Death, Black_Death, Black_Death, Black_Death
  **F1:** 0.800  **Recall@4:** 1.000

- **Q:** What supplanted Frame Relay and X.25 
  **Gold:** supplanted by the Internet Protocol (IP) at the network layer, and the Asynchronous Transfer Mode (ATM) and or versions of Multi-Protocol Label Switching | Internet Protocol (IP) | Internet Protocol
  **Pred:** Internet Protocol (IP), Asynchronous Transfer Mode (ATM), and versions of Multi-Protocol Label Switching (
  **Retrieved titles:** Packet_switching, Packet_switching, Packet_switching, Packet_switching
  **F1:** 0.788  **Recall@4:** 1.000

- **Q:** Contracts must be designed to ensure what?
  **Gold:** that each side is capable of performing the obligations set out | that each side is capable of performing the obligations set out | each side is capable of performing the obligations set out
  **Pred:** Contracts must ensure each side is capable of performing obligations
  **Retrieved titles:** Construction, Construction, Construction, Construction
  **F1:** 0.737  **Recall@4:** 1.000


### Partial Overlap (3 examples shown)

- **Q:** What do ctenophores have that no other animals have?
  **Gold:** colloblasts | colloblasts | colloblasts
  **Pred:** sticky colloblasts for capturing prey
  **Retrieved titles:** Ctenophora, Ctenophora, Ctenophora, Ctenophora
  **F1:** 0.333  **Recall@4:** 1.000

- **Q:** What delimits the delta of the Rhine in the west?
  **Gold:** Alter Rhein | Alter Rhein | Alter Rhein | Alter Rhein
  **Pred:** The Alter Rhein ("Old Rhine") delimits the delta of the Rhine in the west
  **Retrieved titles:** Rhine, Rhine, Rhine, Rhine
  **F1:** 0.333  **Recall@4:** 1.000

- **Q:** What does the lobata have pair of?
  **Gold:** lobes | lobes | lobes
  **Pred:** The lobata has a pair of lobes
  **Retrieved titles:** Ctenophora, Ctenophora, Oxygen, Ctenophora
  **F1:** 0.333  **Recall@4:** 1.000


### Content Error (3 examples shown)

- **Q:** How many counties initially made up the definition of southern California?
  **Gold:** seven | seven | seven
  **Pred:** The context does not provide a specific number of counties initially making up southern California according to the historical attempts for separate statehood or territorial status
  **Retrieved titles:** Southern_California, Southern_California, Southern_California, Southern_California
  **F1:** 0.000  **Recall@4:** 1.000

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


## Dataset: hotpot_qa  Chunker: recursive_256

### Retrieval Failure (0 examples shown)

_No examples in this bucket._

### Formatting Mismatch (3 examples shown)

- **Q:** Pacific Mozart Ensemble performed which German composer's Der Lindberghflug in 2002?
  **Gold:** Kurt Julian Weill
  **Pred:** Kurt Weill
  **Retrieved titles:** The Flight Across the Ocean, Pacific Mozart Ensemble, Pacific Mozart Ensemble, Martin Boykan
  **F1:** 0.800  **Recall@4:** 0.500

- **Q:** What Kentucky county has a population of 60,316 and features the Lake Louisvilla neighborhood?
  **Gold:** Oldham County
  **Pred:** Oldham County, Kentucky
  **Retrieved titles:** Lake Louisvilla, Louisville, Kentucky County, Virginia, Oldham County, Kentucky, Casey County, Kentucky
  **F1:** 0.800  **Recall@4:** 1.000

- **Q:** What commentator and author questioned the validity of CORE's non-profit status?
  **Gold:** Rachel Anne Maddow
  **Pred:** Rachel Maddow
  **Retrieved titles:** Center for Organizational Research and Education, Brown Association for Cooperative Housing, American Bladesmith Society, York City War Memorial
  **F1:** 0.800  **Recall@4:** 0.500


### Partial Overlap (2 examples shown)

- **Q:** Between Pinanga and Alopecurus which genus is more common in temperate regions?
  **Gold:** Alopecurus
  **Pred:** Alopecurus is more common in temperate regions
  **Retrieved titles:** Pinanga, Alopecurus, Alopecurus aequalis, Pison
  **F1:** 0.250  **Recall@4:** 1.000

- **Q:** If the Charhki  Dadri crash was less dangerous than the Tenerife airport disaster, which occured firat? 
  **Gold:** On March 27, 1977, two Boeing 747 passenger jets, KLM Flight 4805 and Pan Am Flight 1736, collided on the runway at Los Rodeos Airport (now Tenerife North Airport)
  **Pred:** The Tenerife airport disaster occurred first
  **Retrieved titles:** Charkhi Dadri mid-air collision, 1977 in aviation, Tenerife airport disaster, Jacob Veldhuyzen van Zanten
  **F1:** 0.121  **Recall@4:** 1.000


### Content Error (3 examples shown)

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

- **Q:** Which mountain is taller, Gasherbrum II or Langtang Ri?
  **Gold:** Gasherbrum II
  **Pred:** taller than Langtang Ri (72
  **Retrieved titles:** Gasherbrum II, Langtang Ri, Gasherbrum I, Gasherbrum III
  **F1:** 0.000  **Recall@4:** 1.000

