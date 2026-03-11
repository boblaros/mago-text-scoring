export interface SampleInput {
  id: string;
  title: string;
  value: string;
}

export const SAMPLE_LIBRARY: SampleInput[] = [
  {
    id: "review",
    title: "Product review",
    value:
      "I picked this up expecting something decent, and honestly it turned out to be one of the most satisfying tools I have used this year. The setup was simple, the interface is clear, and the whole experience feels surprisingly polished.",
  },
  {
    id: "social",
    title: "Social post",
    value:
      "omg this is actually wild, I stayed up way too late finishing it and now I kind of want everyone in the group chat to read it because the ending was so dramatic lol",
  },
  {
    id: "abuse",
    title: "Abuse edge case",
    value:
      "You keep acting like an idiot and it makes every conversation exhausting. Stop trying to spin this like it is normal.",
  },
  {
    id: "academic",
    title: "Academic paragraph",
    value:
      "The article argues that interpretability is not merely a technical afterthought but a necessary condition for trustworthy deployment, especially when automated predictions influence educational, medical, or legal decisions at scale.",
  },
];
