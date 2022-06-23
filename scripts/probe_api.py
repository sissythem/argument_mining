import requests
import json
import sys


try:
    config = json.loads(sys.argv[1])
except IndexError:
    config = {}
try:
    access_token = sys.argv[2]
except IndexError:
    access_token = "123"

try:
    host_url = sys.argv[3]
    if host_url == "ncsr":
        host_url = "143.233.226.60"
except IndexError:
    host_url = 'localhost'

try:
    modelid = sys.argv[4]
except IndexError:
    modelid = 'modelid'

try:
    port = sys.argv[5]
except IndexError:
    port = '8080'

txt = """
«Μα θα πρέπει να έχουμε ευαισθησία σε αυτούς που έχουν 300.000 ευρώ και 200.000 ευρώ εισόδημα και απολαμβάνουν παροχές από τον ΟΑΕΔ;
«Την αντιμετώπιση του φαινομένου της κατάχρησης των παροχών στον ΟΑΕΔ, που έχει εντοπιστεί, ώστε τα επιδόματα και οι παροχές να κατευθύνονται σε εκείνους που τα έχουν πραγματικά ανάγκη», θέτει ως βασική προτεραιότητα ο υπουργός Εργασίας και Κοινωνικών Υποθέσεων Κωστής Χατζηδάκης, με το νομοσχέδιο του υπουργείου «Δουλειές ξανά», που πρόκειται να τεθεί σύντομα σε δημόσια διαβούλευση.

Μιλώντας σε εκπομπή του τηλεοπτικού σταθμού Σκάι και ερωτηθείς για τους ανέργους που λαμβάνουν παροχές από τον ΟΑΕΔ, ενώ έχουν πολύ υψηλά εισοδήματα, ο κ. Χατζηδάκης απάντησε ότι στόχος είναι να μπει τέλος σε αυτά τα φαινόμενα. «Θέλουμε να βάλουμε τέλος σε αυτά τα φαινόμενα και μου λένε ότι δεν έχω κοινωνική ευαισθησία. Μα θα πρέπει να έχουμε ευαισθησία σε αυτούς που έχουν 300.000 ευρώ και 200.000 ευρώ εισόδημα και απολαμβάνουν παροχές από τον ΟΑΕΔ; Να στηρίξουμε τους μακροχρόνια ανέργους ναι, όχι όμως να συγκαλύπτουμε τους απατεώνες και αυτούς που κάνουν καταχρήσεις» ανέφερε χαρακτηριστικά ο υπουργός Εργασίας, προσθέτοντας: «Θέλουμε να δώσουμε προτεραιότητα στους μακροχρόνια ανέργους. Γι' αυτό, τους δίνουμε πριμ 300 ευρώ, για να εγγραφούν στο ψηφιακό μητρώο και να καταρτίσουν ατομικό ψηφιακό σχέδιο δράσης». Συμπληρωματικά, ο κ. Χατζηδάκης σημείωσε ότι αυτό που παρατηρείται είναι ότι, ενώ η ανεργία μειώνεται, ο αριθμός των ανέργων που είναι εγγεγραμμένοι στο μητρώο του ΟΑΕΔ παραμένει αμετάβλητος.

Παράλληλα, στο πλαίσιο της συνέντευξής του, υπογράμμισε ότι, για πρώτη φορά, θεσπίζεται το επίδομα εργασίας για τους επιδοτούμενους ανέργους που βρίσκουν δουλειά. Συγκεκριμένα, διευκρίνισε ότι, όσοι βρίσκουν δουλειά, κατά την περίοδο καταβολής του επιδόματος ανεργίας, θα συνεχίσουν να λαμβάνουν το 50% του επιδόματος ανεργίας, μέχρι την προβλεπόμενη ημερομηνία λήξης της περιόδου καταβολής του.

Μεταξύ άλλων, ο κ. Χατζηδάκης αναφέρθηκε και σε μία ακόμη διάταξη του νομοσχεδίου, η οποία προβλέπει τη διαγραφή από το μητρώο ανέργων, μετά από τρεις αρνήσεις κατάλληλων θέσεων εργασίας. «Εάν ο ΟΑΕΔ σου δίνει τρεις δουλειές αντίστοιχες με το προφίλ σου, τις ικανότητές σου και κοντά στον τόπο κατοικίας σου και αρνηθείς, βγαίνεις από το μητρώο. Πρόκειται για ένα δίκαιο μέτρο, που έπρεπε να εφαρμοστεί από καιρό» διευκρίνισε ο αρμόδιος υπουργός.

Αναφορικά με τον κατώτατο μισθό, ο κ. Χατζηδάκης δήλωσε ότι, σε περίπου 45 ημέρες, θα υπάρξει τοποθέτηση του υπουργού Εργασίας στο υπουργικό συμβούλιο και σχετική απόφαση του υπουργικού συμβουλίου. «Παραμένει η δέσμευση του πρωθυπουργού ότι η αύξηση του κατώτατου μισθού θα είναι σημαντική. Έχουμε δημιουργήσει μέσα σε δυσκολίες τα προηγούμενα δυόμισι χρόνια τις προϋποθέσεις πάνω σε υγιείς και γερές βάσεις, ώστε να υπάρξει μία σημαντική αύξηση του κατώτατου μισθού. Αυτό έγινε, χάρη στη φορολογική, δημοσιονομική, εργασιακή και ασφαλιστική πολιτική της κυβέρνησης, που επέτρεψε να πάει μπροστά η οικονομία και να έχουμε και μείωση της ανεργίας κατά τέσσερις ποσοστιαίες μονάδες μέσα στην πανδημία. Αποδεικνύουμε στην πράξη ότι έχουμε μία φιλεργατική πολιτική. Όταν ωθείς μπροστά την οικονομία, κερδίζουν και οι εργαζόμενοι» τόνισε ο υπουργός Εργασίας, συμπληρώνοντας ότι οι ασφαλιστικές εισφορές έχουν ήδη μειωθεί κατά τέσσερις μονάδες, που και «αυτό βοηθάει να πάρει μπροστά η οικονομία».

Για το ύψος της αύξησης του κατώτατου μισθού, ο κ. Χατζηδάκης απάντησε ότι αυτή είναι η άσκηση που πρέπει να λύσουμε, λαμβάνοντας υπόψη την ανταγωνιστικότητα, την παραγωγικότητα, τις αντοχές της οικονομίας.

Παράλληλα, ο υπουργός Εργασίας έδωσε έμφαση και στο ζήτημα της αντιμετώπισης των εκκρεμών συντάξεων. Όπως είπε, το 2021, καταγράφηκε αύξηση κατά 83% στην έκδοση συντάξεων σε σχέση με το 2019, κάνοντας λόγο για ρεκόρ όλων των εποχών.

Αναφερόμενος στο θέμα της ακρίβειας, ο κ. Χατζηδάκης επεσήμανε ότι το πρόβλημα της ακρίβειας είναι πανευρωπαϊκό. «Προφανώς, ακούμε την ανησυχία και τα προβλήματα του κόσμου και θα προσπαθήσουμε να κάνουμε ό,τι περισσότερο γίνεται ενός των αντοχών της οικονομίας και του προϋπολογισμού» υπογράμμισε ο υπουργός Εργασίας και τόνισε ότι απαιτείται παρέμβαση σε ευρωπαϊκό επίπεδο για τις τιμές ενέργειας.

Στο πλαίσιο αυτό, υπενθύμισε ότι η κυβέρνηση έχει δώσει πάνω από 2 δισ. ευρώ για τις επιδοτήσεις σε ρεύμα και φυσικό αέριο και η ΔΕΗ έχει δώσει 800 εκατ. ευρώ από την κερδοφορία της, που θα είναι οριακή. «Αν δεν είχαμε σώσει τη ΔΕΗ το 2019, πού θα έβρισκε αυτά τα χρήματα;» διερωτήθηκε ο κ. Χατζηδάκης, ενώ προσέθεσε ότι, αν δεν είχε γίνει η παρέμβαση με το λιγνίτη, το ρεύμα θα ήταν ακόμα ακριβότερο.

Συμπερασματικά, ο υπουργός Εργασίας δήλωσε ότι τα νέα μέτρα για την ακρίβεια, που θα ανακοινωθούν τις επόμενες ημέρες, θα είναι δίκαια και ισορροπημένα. Τόνισε επίσης ότι θα εξαντληθούν όλες οι δυνατότητες, για να στηριχθούν οι ευάλωτοι κατά προτεραιότητα. «Δεν υπάρχει η δυνατότητα ούτε στην Ελλάδα ούτε σε καμία άλλη χώρα της ΕΕ να εξαφανιστεί το πρόβλημα. Δεν είμαστε θαυματοποιοί» σχολίασε ο κ. Χατζηδάκης, ο οποίος επανέλαβε ότι η έμφαση θα δοθεί στους πιο φτωχούς, που συμπιέζονται περισσότερο.

Σχετικά Tags
ΟΑΕΔΚωστής Χατζηδάκης
"""

txt_english = """
Across this shattered country, testimony is being gathered and evidence collated for a goal that may only come long after the guns fall silent - international justice.

The pursuit of it is being led by Ukraine's Prosecutor General, Iryna Venediktova, appointed two years ago as the first woman to hold the office.

She watched the exhumation of another mass grave this week, beneath a gold-domed church in Bucha, where the darkest of sins were discovered - 10 victims this time, some completely charred. Their remains were placed into body bags and taken off for an attempt at identification.

As she stood at the edge of the deep pit, she told me more than 6,000 cases of war crimes have already been opened.

"A lot of people speak about the genocide of the Ukrainian people - and we actually have grounds to talk about genocide," she said. "Vladimir Putin is the president of the aggressor country killing civilians here in Ukraine. He's responsible."

The Kremlin continues to deny such allegations.
"""


url = "https://www.news.gr/politikh/article/2862428/chatzidakis-me-to-doulies-xana-beni-telos-sta-fenomena-katachrisis-ton-parochon-ston-oaed.html"
# This url causes tokenization issues
# url = "https://www.efsyn.gr/politiki/exoteriki-politiki/332706_dendias-i-symfonia-gia-apostratiotikopoiisi-den-aforoyse-tin",
urls = [
    "https://tvxs.gr/news/i-apopsi-mas/o-isyxos-thanatos-tis-pandimias",
    "https://tvxs.gr/news/egrapsan-eipan/ena-dakry-gia-tin-eleytheria-toy-typoy",
    "https://www.metaforespress.gr/aeroporika/2025-%CF%83%CE%B5-%CE%BB%CE%B5%CE%B9%CF%84%CE%BF%CF%85%CF%81%CE%B3%CE%AF%CE%B1-%CF%84%CE%BF-%CE%BD%CE%AD%CE%BF-%CE%B1%CE%B5%CF%81%CE%BF%CE%B4%CF%81%CF%8C%CE%BC%CE%B9%CE%BF-%CF%83%CF%84%CE%BF-%CE%BA/"
]
payload = {
    "links": urls,
    "model_id": modelid
    # "model_id": 'configured'
    # "texts": [txt_english]
}

payload["config"] = config
response = requests.post(
    f"http://{host_url}:{port}/predict",
    json=payload,
    params={"access_token": access_token}
)
res = json.loads(response.content)
print(res)
