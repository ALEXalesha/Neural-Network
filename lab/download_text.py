import wikipedia
import os

cache_file = "books_cache.txt"

TOPICS = [
    # Уже были (45 статей)
    "Russia", "Moscow", "Saint Petersburg", "History of Russia",
    "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
    "World War II", "Solar System", "Universe",
    "Literature", "Alexander Pushkin", "Leo Tolstoy", "Fyodor Dostoevsky",
    "Economics", "Technology", "Computer", "Internet", "Artificial intelligence",
    "Art", "Painting", "Architecture",
    "Climate", "Evolution",
    "Philosophy", "Psychology", "Medicine",
    "Football", "Olympic Games", "Theatre",
    "Europe", "Africa", "Americas",
    "French Revolution", "Napoleon", "Peter the Great",
    "Theory of relativity", "DNA", "Genetics",
    "Ecology", "Geology", "Geography", "Sociology",

    # Новые — ещё ~100 тем
    "Vladimir Putin", "Joseph Stalin", "Lenin", "Catherine the Great",
    "World War I", "Cold War", "Soviet Union", "Roman Empire",
    "Ancient Greece", "Byzantine Empire", "Mongol Empire", "Ottoman Empire",
    "United States", "China", "Germany", "France", "United Kingdom",
    "Japan", "India", "Brazil", "Ukraine", "Kazakhstan",
    "Albert Einstein", "Isaac Newton", "Charles Darwin", "Galileo Galilei",
    "Nikola Tesla", "Marie Curie", "Stephen Hawking", "Carl Sagan",
    "Black hole", "Galaxy", "Mars", "Jupiter", "Moon",
    "Atom", "Electron", "Nuclear physics", "Thermodynamics", "Electromagnetism",
    "Cell biology", "Human brain", "Heart", "Cancer", "Vaccination",
    "Python programming language", "Machine learning", "Neural network",
    "Blockchain", "Smartphone", "Satellite", "Electric car",
    "Climate change", "Ocean", "Rainforest", "Volcano", "Earthquake",
    "Dinosaur", "Whale", "Eagle", "Tiger", "Elephant",
    "Plato", "Aristotle", "Immanuel Kant", "Friedrich Nietzsche",
    "Sigmund Freud", "Carl Jung", "Noam Chomsky",
    "William Shakespeare", "Dante Alighieri", "Homer", "Franz Kafka",
    "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach",
    "Leonardo da Vinci", "Michelangelo", "Pablo Picasso",
    "Football (soccer)", "Basketball", "Tennis", "Chess", "Swimming",
    "Democracy", "Communism", "Capitalism", "Liberalism",
    "Christianity", "Islam", "Buddhism", "Judaism",
    "Agriculture", "Industrial Revolution", "Printing press", "Steam engine",
    "Trade", "Bank", "Stock market", "Inflation",
    "Human rights", "United Nations", "European Union",
    "Cinema", "Animation", "Video game", "Photography",
    "Algebra", "Geometry", "Calculus", "Statistics", "Logic",
    "Linguistics", "Anthropology", "Archaeology", "Political science",
    "Climate", "Weather", "River", "Mountain", "Desert",
    "Nutrition", "Sport", "Yoga", "Marathon",
    "Antibiotic", "Surgery", "Genetics", "Pandemic",
    "Telescope", "Microscope", "Laser", "Radio", "Television",
]

# Убираем дубликаты
TOPICS = list(dict.fromkeys(TOPICS))

wikipedia.set_lang("ru")
text = ""
ok, fail = 0, 0

print(f"Скачиваем {len(TOPICS)} статей...\n")

for topic in TOPICS:
    try:
        page = wikipedia.page(topic, auto_suggest=True)
        text += page.content + "\n\n"
        ok += 1
        if ok % 10 == 0:
            print(f"  [{ok:3d}] Символов накоплено: {len(text):,}")
    except Exception:
        fail += 1

print(f"\nУспешно: {ok} | Ошибок: {fail}")
print(f"Итого символов: {len(text):,} ({len(text)//1000} КБ)")

with open(cache_file, 'w', encoding='utf-8') as f:
    f.write(text)

print("Сохранено в books_cache.txt")
