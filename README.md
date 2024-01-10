# Sorunun-Cevabı-Nerede
- Projemizde soruların hangi dokümanlarda cevaplandığını bulabilmek için TF-IDF, BERT ve mGPT gibi modellerden soru ve metinlerin vektör temsillerini aldıktan sonra bunlar arasındaki kosinüs benzerliğini hesaplıyoruz.

# Türkçe Soru Cevap Veri Kümesi
### Giriş
Doğal dil işleme (NLP) alanındaki araştırmalar, özellikle farklı dillerdeki veri kümesi çeşitliliğinin artmasıyla son yıllarda büyük ilerlemeler kaydetti. Ancak, Türkçe gibi bazı dillerde herkese açık ve kaliteli soru-cevap metin veri kümelerinin eksikliği, dilimizde NLP uygulamalarının gelişimini sınırlamaktadır. Bu durum, araştırmamızda kullandığımız veri kümesini oluşturma sürecinde temel motivasyon kaynağımız oldu.

### Veri Kümesi Oluşturma Süreci
Projemiz için gerekli Türkçe soru-cevap veri kümesini oluştururken iki ana kaynaktan yararlandık:

- Türkçe Wikipedia Sayfaları: Wikipedia, geniş kapsamlı ve çoğu zaman güvenilir bilgi içeren, çeşitli konularda yazılmış metinleri barındıran bir kaynaktır. Bu platform, doğal dil işleme uygulamaları için zengin bir metin kaynağı sunar. Wikipedia’dan seçilen metinleri işleyerek projemizin temel veri kümesini oluşturduk.

- Stanford Question Answering Dataset (SQuAD): Stanford Üniversitesi’nin İngilizce olarak hazırlamış olduğu SQuAD veri kümesinde bulunan bazı metinleri Türkçe diline çevirerek veri kümemize dahil ettik. Çeviri sürecinde, metinlerin doğallığını ve anlam bütünlüğünü korumaya özen gösterdik.

### Sonuç
Sonuç olarak, bu iki farklı kaynaktan elde edilen metinleri birleştirerek ufak çaplı, Türkçe NLP uygulamaları için değerli bir kaynak oluşturduk. Bu veri kümesi, Türkçe metin işleme teknolojilerinin gelişimi için ufak bir adım oluşturmakta ve gelecekteki NLP çalışmalarına katkı sağlamayı hedeflemektedir.