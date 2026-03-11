export function getCompactDomainLabel(
  domain: { domain: string; display_name: string },
): string {
  if (domain.domain === "complexity") {
    return "Complexity";
  }
  if (domain.domain === "abuse") {
    return "Cyberbullying";
  }
  return domain.display_name;
}
