use burn::data::dataset::{Dataset, InMemDataset};
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgNewsItem {
    pub id:usize,
    pub keyword:String,
    pub location:String,
    pub text: String,
    pub target: usize, 
}
pub struct DiabetesDataset {
    dataset: InMemDataset<AgNewsItem>,
}
impl DiabetesDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        let dataset = InMemDataset::from_csv("./datasets/nlp-getting-started/train.csv").unwrap();
        let dataset = Self { dataset };

        Ok(dataset)
    }
}
impl Dataset<AgNewsItem> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<AgNewsItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
pub fn main(){
    let a= DiabetesDataset::new().unwrap();
    println!("{:?}",a.get(0));

}